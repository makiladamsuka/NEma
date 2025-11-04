import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot access camera")
    exit()

# Waving detection memory (stores last 10 wrist x positions)
history = deque(maxlen=10)
wave_detected = False

# Pose model
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        frame_h, frame_w, _ = frame.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Right wrist & shoulder
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            wrist_x = wrist.x
            wrist_y = wrist.y

            # Add new wrist position to history
            history.append(wrist_x)

            # Check if hand is raised above shoulder
            hand_up = wrist_y < shoulder.y

            # Detect waving based on horizontal movement range
            if len(history) == history.maxlen and hand_up:
                movement = max(history) - min(history)
                if movement > 0.15:  # waving threshold
                    wave_detected = True
                else:
                    wave_detected = False

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display text
            if wave_detected:
                cv2.putText(frame, "👋 Waving Detected!", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No Wave Detected", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "No Person Detected", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Waving Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
