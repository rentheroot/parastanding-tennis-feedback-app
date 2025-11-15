import cv2
import mediapipe as mp

# Start video feed on python
class VideoFeed:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

    def start_feed(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

    def trace_body_pos(self):

        # Create window
        cv2.namedWindow('Tennis Tracer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Tennis Tracer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Draw on feed
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                        self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                        self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                cv2.imshow('Tennis Tracer', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
        self.cap.release()
        cv2.destroyAllWindows()

vid_feed = VideoFeed()
vid_feed.start_feed()
vid_feed.trace_body_pos()
