# main_with_detector_no_wrist.py
"""
Full live detection pipeline (no wrist-speed fallback).

- Loads model_pipeline.pkl and model_meta.json
- Uses moving-average of recent probabilities to make decisions
- Uses CONSEC_POS_N consecutive moving-avg positives to START clip
- Uses CONSEC_NEG_N consecutive moving-avg negatives to STOP clip
- Saves annotated + raw video clips and a sqlite DB with pose rows for each clip
- No wrist-speed heuristics or fallback logic anywhere

Adjust top-level variables as needed (no argparse).
"""

import os
import time
import json
import cv2
import sqlite3
import threading
import queue
import datetime
from collections import deque, Counter
import numpy as np
import mediapipe as mp
from joblib import load

# ----------------- USER / HYPERPARAMETERS -----------------
MODEL_PIPELINE = "model_pipeline.pkl"
MODEL_META = "model_meta.json"

WINDOW_LEN = 1.0           # seconds used to aggregate features (should match training)
SAMPLE_INTERVAL = 0.1      # seconds between evaluations (0.1 -> 10Hz)
MOVING_AVG_WINDOW = 5      # how many recent probs to average for smoothing
PROB_THRESHOLD = 0.30      # moving-average probability threshold to consider a positive
CONSEC_POS_N = 1           # consecutive moving-avg positives to START clip
CONSEC_NEG_N = 3           # consecutive moving-avg negatives to STOP clip
SAVE_CLIP_VIDEO_FPS = 20   # fps of saved clip video
CLIP_DIR = "clips/detected/"
ANNOTATED_DIR = os.path.join(CLIP_DIR, "annotated")
RAW_DIR = os.path.join(CLIP_DIR, "raw")
DB_DIR = os.path.join(CLIP_DIR, "dbs")

USERNAME_FOR_DB = "Renee"  # used by AsyncDBWriter to name main DB
MAX_BUFFER_SECONDS = 12.0  # how many seconds to keep in memory for pre-roll
# --------------------------------------------------------

for p in (CLIP_DIR, ANNOTATED_DIR, RAW_DIR, DB_DIR):
    os.makedirs(p, exist_ok=True)

# ---------------- utilities ----------------
def timestamp_to_str(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y%m%d_%H%M%S_%f")

# ---------------- AsyncDBWriter ----------------
class AsyncDBWriter(threading.Thread):
    def __init__(self, username):
        super().__init__(daemon=True)
        self.db = SQLManager(username)
        self.db.connect_to_unfiltered_db()
        self.q = queue.Queue()
        self.running = True

    def run(self):
        while self.running:
            try:
                batch = self.q.get(timeout=1)
            except queue.Empty:
                continue

            if batch is None:
                break

            for table, values in batch.items():
                self.db.cur_unfiltered.executemany(
                    f"INSERT INTO {table} VALUES(?, ?, ?, ?, ?, ?)",
                    values
                )
            self.db.conn_unfiltered.commit()

    def submit(self, pose_data, right_hand_data, left_hand_data):
        self.q.put({
            "body_pos": pose_data,
            "right_hand_pos": right_hand_data,
            "left_hand_pos": left_hand_data
        })

    def stop(self):
        self.running = False
        self.q.put(None)

    def write_clip_db(self, rows, out_path):
        conn = sqlite3.connect(out_path)
        cur = conn.cursor()
        cur.execute(''' CREATE TABLE IF NOT EXISTS body_pos(
                            landmark VARCHAR(20), x REAL, y REAL, z REAL, visibility REAL, current_time TIMESTAMP) ''')
        cur.executemany("INSERT INTO body_pos VALUES(?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()

# ---------------- support classes ----------------
class BodySelection:
    def __init__(self, config=None):
        self.main_pose_names = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 
                                'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 
                                'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 
                                'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 
                                'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
                                'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 
                                'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 
                                'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 
                                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 
                                'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 
                                'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

class SQLManager:
    def __init__(self, username):
        self.username = username

    def connect_to_unfiltered_db(self):
        try:
            self.conn_unfiltered = sqlite3.connect(
                f"{self.username}_unfiltered_data.db", check_same_thread=False
            )
            self.cur_unfiltered = self.conn_unfiltered.cursor()
            self.cur_unfiltered.execute(''' CREATE TABLE IF NOT EXISTS body_pos(
                                        landmark VARCHAR(20), 
                                        x REAL, 
                                        y REAL, 
                                        z REAL, 
                                        visibility REAL, 
                                        current_time TIMESTAMP) ''')
            self.conn_unfiltered.commit()
        except sqlite3.OperationalError as e:
            print("Failed to open database:", e)

# ---------------- main video feed with detector (no wrist fallback) ----------------
class VideoFeedNoWrist:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Start async DB writer
        self.db_writer = AsyncDBWriter(USERNAME_FOR_DB)
        self.db_writer.start()

        # Load model and meta
        if not os.path.exists(MODEL_PIPELINE) or not os.path.exists(MODEL_META):
            raise RuntimeError("Model or meta not found. Run train_model.py first.")
        self.pipeline = load(MODEL_PIPELINE)
        with open(MODEL_META, "r") as f:
            self.meta = json.load(f)
        self.landmarks_order = self.meta["landmarks_order"]
        self.feat_per_landmark = self.meta.get("feature_per_landmark", ["x","y","z","visibility"])
        self.vec_len = self.meta["feature_vector_length"]

        # Buffers
        self.pose_buffer = deque()  # (ts, [ (landmark,x,y,z,vis), ... ])
        self.frame_buffer_raw = deque()
        self.frame_buffer_annot = deque()
        self.max_buffer_seconds = MAX_BUFFER_SECONDS

        # Detector state
        self.prob_history = deque(maxlen=MOVING_AVG_WINDOW)
        self.recording = False
        self.record_start_ts = None
        self.record_frames_raw = []
        self.record_frames_annot = []
        self.record_pose_rows = []

    def start_feed(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

    def _prune_old_buffers(self):
        cutoff = time.time() - self.max_buffer_seconds
        while self.pose_buffer and self.pose_buffer[0][0] < cutoff:
            self.pose_buffer.popleft()
        while self.frame_buffer_raw and self.frame_buffer_raw[0][0] < cutoff:
            self.frame_buffer_raw.popleft()
        while self.frame_buffer_annot and self.frame_buffer_annot[0][0] < cutoff:
            self.frame_buffer_annot.popleft()

    def _aggregate_window_vector(self, now_ts):
        """
        Aggregate mean-of-landmarks over WINDOW_LEN seconds ending at now_ts.
        Returns 1xD numpy vector or None if no data.
        """
        start_ts = now_ts - WINDOW_LEN
        rows = []
        # iterate from right (newest) to oldest until start cutoff
        for ts, lm_list in reversed(self.pose_buffer):
            if ts < start_ts:
                break
            rows.append((ts, lm_list))
        if not rows:
            return None

        accum = {L: [] for L in self.landmarks_order}
        for ts, lm_list in rows:
            for L, x, y, z, vis in lm_list:
                if L in accum:
                    accum[L].append((x, y, z, vis))
        vec = []
        for L in self.landmarks_order:
            vals = accum[L]
            if vals:
                arr = np.array(vals, dtype=float)
                mean_vals = np.nanmean(arr, axis=0)
                vec.extend([float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2]), float(mean_vals[3])])
            else:
                vec.extend([np.nan, np.nan, np.nan, np.nan])
        v = np.array(vec, dtype=float)
        if np.isnan(v).all():
            return None
        # impute NaNs with global mean (simple)
        col_mean = np.nanmean(v)
        inds = np.isnan(v)
        v[inds] = col_mean
        return v.reshape(1, -1)

    def _predict_now(self):
        now_ts = time.time()
        vec = self._aggregate_window_vector(now_ts)
        if vec is None:
            # no data → return probability 0 and negative decision
            buffer_size = len(self.pose_buffer)
            print(f"[WARN] No aggregated vector (pose_buffer size: {buffer_size})")
            return 0.0, False, {"moving_avg": float(np.mean(self.prob_history)) if len(self.prob_history) else 0.0}

        proba = float(self.pipeline.predict_proba(vec)[0][1])  # probability of positive class (1)
        # moving average
        self.prob_history.append(proba)
        moving_avg = float(np.mean(self.prob_history)) if len(self.prob_history) else 0.0
        positive = (moving_avg >= PROB_THRESHOLD)
        return proba, positive, {"moving_avg": moving_avg}

    def _start_recording(self, start_ts):
        print("=== START RECORDING at", start_ts)
        self.recording = True
        self.record_start_ts = start_ts
        self.record_frames_raw = []
        self.record_frames_annot = []
        self.record_pose_rows = []

    def _stop_and_save_recording(self, end_ts):
        print("=== STOP RECORDING at", end_ts)
        self.recording = False
        basename = f"swing_{timestamp_to_str(self.record_start_ts)}_{timestamp_to_str(end_ts)}"

        # save raw video
        if self.record_frames_raw:
            h, w = self.record_frames_raw[0][1].shape[:2]
            raw_path = os.path.join(RAW_DIR, f"{basename}_raw.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(raw_path, fourcc, SAVE_CLIP_VIDEO_FPS, (w, h))
            for ts, frame in self.record_frames_raw:
                out.write(frame)
            out.release()
            print("Saved raw video:", raw_path)

        # save annotated video
        if self.record_frames_annot:
            h, w = self.record_frames_annot[0][1].shape[:2]
            ann_path = os.path.join(ANNOTATED_DIR, f"{basename}_annot.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(ann_path, fourcc, SAVE_CLIP_VIDEO_FPS, (w, h))
            for ts, frame in self.record_frames_annot:
                out.write(frame)
            out.release()
            print("Saved annotated video:", ann_path)

        # save clip DB (pose rows)
        if self.record_pose_rows:
            db_path = os.path.join(DB_DIR, f"{basename}_poses.sqlite")
            self.db_writer.write_clip_db(self.record_pose_rows, db_path)
            print("Saved clip DB:", db_path)

        # clear buffers
        self.record_frames_raw = []
        self.record_frames_annot = []
        self.record_pose_rows = []
        self.record_start_ts = None

    def trace_body_pos(self):
        # fullscreen window (same as previous)
        cv2.namedWindow('Tennis Tracer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Tennis Tracer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5) as holistic:

            last_eval = 0.0
            consecutive_pos = 0
            consecutive_neg = 0
            landmarks_checked = False

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_ts = time.time()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # build annotated copy
                annotated = image_bgr.copy()
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(annotated, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                # collect pose rows for this frame (main body)
                frame_pose_rows = []
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        name = self.mp_holistic.PoseLandmark(idx).name
                        frame_pose_rows.append((name, float(landmark.x), float(landmark.y), float(landmark.z), float(landmark.visibility), datetime.datetime.fromtimestamp(frame_ts).strftime("%Y-%m-%d %H:%M:%S.%f")))

                # DEBUG: Check landmark name consistency on first frame
                if not landmarks_checked and frame_pose_rows:
                    detected_names = sorted([r[0] for r in frame_pose_rows])
                    expected_names = sorted(self.landmarks_order)
                    landmarks_checked = True
                    print(f"[INFO] Expected landmarks from model: {expected_names}")
                    print(f"[INFO] Detected landmarks from MediaPipe: {detected_names}")
                    if set(detected_names) != set(expected_names):
                        print(f"[WARN] Landmark mismatch! Expected {len(expected_names)}, got {len(detected_names)}")
                        print(f"[WARN] Missing in detection: {set(expected_names) - set(detected_names)}")
                        print(f"[WARN] Extra in detection: {set(detected_names) - set(expected_names)}")
                    else:
                        print(f"[OK] Landmarks match!")


                # right & left hands (kept for DB submission, but not used for detection aggregation unless model includes them)
                frame_right_rows = []
                if results.right_hand_landmarks:
                    for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                        name = "R_" + self.mp_holistic.HandLandmark(idx).name
                        frame_right_rows.append((name, float(landmark.x), float(landmark.y), float(landmark.z), float(landmark.visibility), datetime.datetime.fromtimestamp(frame_ts).strftime("%Y-%m-%d %H:%M:%S.%f")))
                frame_left_rows = []
                if results.left_hand_landmarks:
                    for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                        name = "L_" + self.mp_holistic.HandLandmark(idx).name
                        frame_left_rows.append((name, float(landmark.x), float(landmark.y), float(landmark.z), float(landmark.visibility), datetime.datetime.fromtimestamp(frame_ts).strftime("%Y-%m-%d %H:%M:%S.%f")))

                # add to buffers
                if frame_pose_rows:
                    simple_rows = [(r[0], r[1], r[2], r[3], r[4]) for r in frame_pose_rows]
                    self.pose_buffer.append((frame_ts, simple_rows))

                self.frame_buffer_raw.append((frame_ts, frame.copy()))
                self.frame_buffer_annot.append((frame_ts, annotated.copy()))
                self._prune_old_buffers()

                # periodic evaluation
                if frame_ts - last_eval >= SAMPLE_INTERVAL:
                    proba, positive, diag = self._predict_now()
                    last_eval = frame_ts
                    moving_avg = diag.get("moving_avg", 0.0) if diag else 0.0

                    # debug print (safe formatting)
                    print(f"[DET] t={frame_ts:.2f} prob={proba:.3f} mov_avg={moving_avg:.3f} -> positive={positive}")

                    if positive:
                        consecutive_pos += 1
                        consecutive_neg = 0
                    else:
                        consecutive_pos = 0
                        consecutive_neg += 1

                    # start recording on consecutive positives
                    if (not self.recording) and (consecutive_pos >= CONSEC_POS_N):
                        self._start_recording(frame_ts)
                        # include recent pre-roll
                        start_cutoff = frame_ts - WINDOW_LEN - 0.2
                        for ts_f, fr in list(self.frame_buffer_raw):
                            if ts_f >= start_cutoff:
                                self.record_frames_raw.append((ts_f, fr.copy()))
                        for ts_f, fr in list(self.frame_buffer_annot):
                            if ts_f >= start_cutoff:
                                self.record_frames_annot.append((ts_f, fr.copy()))
                        for ts_p, lm_list in list(self.pose_buffer):
                            if ts_p >= start_cutoff:
                                for L, x, y, z, vis in lm_list:
                                    self.record_pose_rows.append((L, x, y, z, vis, datetime.datetime.fromtimestamp(ts_p).strftime("%Y-%m-%d %H:%M:%S.%f")))

                    # if recording, keep appending and stop on consecutive negs
                    if self.recording:
                        self.record_frames_raw.append((frame_ts, frame.copy()))
                        self.record_frames_annot.append((frame_ts, annotated.copy()))
                        for L, x, y, z, vis in (simple_rows if frame_pose_rows else []):
                            self.record_pose_rows.append((L, x, y, z, vis, datetime.datetime.fromtimestamp(frame_ts).strftime("%Y-%m-%d %H:%M:%S.%f")))

                        if consecutive_neg >= CONSEC_NEG_N:
                            end_ts = frame_ts
                            self._stop_and_save_recording(end_ts)
                            consecutive_pos = 0
                            consecutive_neg = 0

                # show annotated frame
                cv2.imshow('Tennis Tracer', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.db_writer.stop()

# ---------------- run ----------------
if __name__ == "__main__":
    vf = VideoFeedNoWrist()
    vf.start_feed(cam_index=0)
    vf.trace_body_pos()
