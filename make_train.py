import cv2
import mediapipe as mp
import numpy as np
import datetime
import sqlite3
import threading
import queue
import time

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

            # batch is: {"table": [rows], ...}
            for table, values in batch.items():
                self.db.cur_unfiltered.executemany(
                    f"INSERT INTO {table} VALUES(?, ?, ?, ?, ?, ?, ?)",  # 7 columns now
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


class BodySelection:
    def __init__(self, config):
        self.config = config
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

        self.right_hand_pose_names = ['R_WRIST', 'R_THUMB_CMC', 'R_THUMB_MCP',
                                'R_THUMB_IP', 'R_THUMB_TIP', 'R_INDEX_FINGER_MCP',
                                'R_INDEX_FINGER_PIP', 'R_INDEX_FINGER_DIP', 'R_INDEX_FINGER_TIP',
                                'R_MIDDLE_FINGER_MCP', 'R_MIDDLE_FINGER_PIP', 'R_MIDDLE_FINGER_DIP',
                                'R_MIDDLE_FINGER_TIP', 'R_RING_FINGER_MCP', 'R_RING_FINGER_PIP',
                                'R_RING_FINGER_DIP', 'R_RING_FINGER_TIP', 'R_PINKY_MCP',
                                'R_PINKY_PIP', 'R_PINKY_DIP', 'R_PINKY_TIP']

        self.left_hand_pose_names = ['L_WRIST', 'L_THUMB_CMC', 'L_THUMB_MCP',
                                     'L_THUMB_IP', 'L_THUMB_TIP', 'L_INDEX_FINGER_MCP',
                                     'L_INDEX_FINGER_PIP', 'L_INDEX_FINGER_DIP', 'L_INDEX_FINGER_TIP',
                                     'L_MIDDLE_FINGER_MCP', 'L_MIDDLE_FINGER_PIP', 'L_MIDDLE_FINGER_DIP',
                                     'L_MIDDLE_FINGER_TIP', 'L_RING_FINGER_MCP', 'L_RING_FINGER_PIP',
                                     'L_RING_FINGER_DIP', 'L_RING_FINGER_TIP', 'L_PINKY_MCP',
                                     'L_PINKY_PIP', 'L_PINKY_DIP', 'L_PINKY_TIP']


class SQLManager:
    def __init__(self, username):
        self.username = username

    def connect_to_unfiltered_db(self):
        try:
            self.conn_unfiltered = sqlite3.connect(
                f"{self.username}_unfiltered_data.db", check_same_thread=False
            )
            print(f"Opened SQLite database successfully.")

            self.cur_unfiltered = self.conn_unfiltered.cursor()

            # Add frame_idx as INTEGER
            self.cur_unfiltered.execute(''' CREATE TABLE IF NOT EXISTS body_pos(
                                        landmark VARCHAR(20), 
                                        x INTEGER, 
                                        y INTEGER, 
                                        z INTEGER, 
                                        visibility,
                                        current_time TIMESTAMP,
                                        frame_idx INTEGER) ''')
            
            self.cur_unfiltered.execute('''CREATE TABLE IF NOT EXISTS right_hand_pos(
                                        landmark VARCHAR(20), 
                                        x INTEGER, 
                                        y INTEGER, 
                                        z INTEGER, 
                                        visibility, 
                                        current_time TIMESTAMP,
                                        frame_idx INTEGER) ''')
            
            self.cur_unfiltered.execute('''CREATE TABLE IF NOT EXISTS left_hand_pos(
                                        landmark VARCHAR(20), 
                                        x INTEGER, 
                                        y INTEGER, 
                                        z INTEGER, 
                                        visibility, 
                                        current_time TIMESTAMP,
                                        frame_idx INTEGER) ''')

        except sqlite3.OperationalError as e:
            print("Failed to open database:", e)


class VideoFeed:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Start async writer
        self.db_writer = AsyncDBWriter("video6")
        self.db_writer.start()

    def start_feed(self):
        self.cap = cv2.VideoCapture('video6.mp4')
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")

    def trace_body_pos(self):

        cv2.namedWindow('Tennis Tracer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Tennis Tracer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        frame_left_hand_keypoints = []
        frame_right_hand_keypoints = []
        frame_pose_keypoints = []

        current_line = 0  # this IS the frame index

        with self.mp_holistic.Holistic(min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5) as holistic:

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                current_time = datetime.datetime.now()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                # Collect main body
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        frame_pose_keypoints.append([
                            self.mp_holistic.PoseLandmark(idx).name,
                            landmark.x, landmark.y, landmark.z,
                            landmark.visibility, current_time,
                            current_line               # <-- frame index added
                        ])

                # Collect right hand
                if results.right_hand_landmarks:
                    for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                        frame_right_hand_keypoints.append([
                            "R_" + self.mp_holistic.HandLandmark(idx).name,
                            landmark.x, landmark.y, landmark.z,
                            landmark.visibility, current_time,
                            current_line               # <-- frame index added
                        ])

                # Collect left hand
                if results.left_hand_landmarks:
                    for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                        frame_left_hand_keypoints.append([
                            "L_" + self.mp_holistic.HandLandmark(idx).name,
                            landmark.x, landmark.y, landmark.z,
                            landmark.visibility, current_time,
                            current_line               # <-- frame index added
                        ])

                # Every 500 frames: offload to DB thread
                if current_line % 500 == 0 and current_line != 0:
                    self.db_writer.submit(
                        frame_pose_keypoints,
                        frame_right_hand_keypoints,
                        frame_left_hand_keypoints
                    )
                    frame_pose_keypoints = []
                    frame_right_hand_keypoints = []
                    frame_left_hand_keypoints = []

                current_line += 1  # increment frame index

                cv2.imshow('Tennis Tracer', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()
        self.db_writer.stop()


# -----------------------------
# Run Program
# -----------------------------
vid = VideoFeed()
vid.start_feed()
vid.trace_body_pos()
