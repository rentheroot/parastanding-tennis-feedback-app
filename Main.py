import cv2
import mediapipe as mp
import numpy as np
import datetime
import sqlite3

# Select body parts for tracking
class BodySelection:

    # config is dict of body parts and whether to include
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

    def identify_excluded(self):
        excluded_main_pose = [pose for pose, include in self.config.items() if pose in self.main_pose_names and include == 0]
        excluded_right_hand = [pose for pose, include in self.config.items() if pose in self.right_hand_pose_names and include == 0]
        excluded_left_hand = [pose for pose, include in self.config.items() if pose in self.left_hand_pose_names and include == 0]
        return (excluded_main_pose, 
                excluded_right_hand,
                excluded_left_hand)

class SQLManager:
    def __init__(self, username):
        self.username = username

    # Connect to unfiltered database and make tables
    def connect_to_unfiltered_db(self):
        try:
            with sqlite3.connect(f"{self.username}_unfiltered_data.db") as self.conn_unfiltered:
                print(f"Opened SQLite database with version {sqlite3.sqlite_version} successfully.")

                # Create cursor for execution
                self.cur_unfiltered = self.con_unfiltered.cursor()

                # Create tables for each group of positions
                self.cur_unfiltered.execute("CREATE TABLE IF NOT EXISTS body_pos(landmark, x, y, z, visibility, current_time)")
                self.cur_unfiltered.execute("CREATE TABLE IF NOT EXISTS right_hand_pos(landmark, x, y, z, visibility, current_time)")
                self.cur_unfiltered.execute("CREATE TABLE IF NOT EXISTS left_hand_pos(landmark, x, y, z, visibility, current_time)")

        except sqlite3.OperationalError as e:
            print("Failed to open database:", e)

    def commit_to_unfiltered_db(self, lines):
        pass


# Start video feed on python
class VideoFeed:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Database init
        self.DB = SQLManager('Renee')
        self.DB.connect_to_unfiltered_db()

    def start_feed(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

    def trace_body_pos(self):

        # Create window
        cv2.namedWindow('Tennis Tracer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Tennis Tracer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # For saving every 50-ish lines of data
        current_line = 0

        # Draw on feed
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                current_time = datetime.datetime.now()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # ------------------------------------
                # If Main landmark points are detected
                # ------------------------------------

                if results.pose_landmarks:

                    frame_pose_keypoints = {}

                    for idx, landmark in enumerate(results.pose_landmarks.landmark):

                        # pose_name : [x, y, z, visibility, timestamp]
                        frame_pose_keypoints[self.mp_holistic.PoseLandmark(idx).name]= [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                            current_time
                        ]

                    # Database write here
                    print(frame_pose_keypoints)

                # -------------------------------------------
                # If Right Hand landmark points are detected
                # -------------------------------------------

                if results.right_hand_landmarks:

                    frame_right_hand_keypoints = {}

                    for idx, landmark in enumerate(results.right_hand_landmarks.landmark):

                        # pose_name : [x, y, z, visibility, timestamp]
                        frame_right_hand_keypoints['R_' + self.mp_holistic.HandLandmark(idx).name]= [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                            current_time
                        ]

                    # Database write here
                    print(frame_right_hand_keypoints)

                # -------------------------------------------
                # If Left Hand landmark points are detected
                # -------------------------------------------

                if results.left_hand_landmarks:

                    frame_left_hand_keypoints = {}

                    for idx, landmark in enumerate(results.left_hand_landmarks.landmark):

                        # pose_name : [x, y, z, visibility, timestamp]
                        frame_left_hand_keypoints['L_' + self.mp_holistic.HandLandmark(idx).name]= [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                            current_time
                        ]

                    # Database write here
                    print(frame_left_hand_keypoints)

                # Write to databases every 50 lines


                cv2.imshow('Tennis Tracer', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                
        self.cap.release()
        cv2.destroyAllWindows()

vid_feed = VideoFeed()
vid_feed.start_feed()
vid_feed.trace_body_pos()
