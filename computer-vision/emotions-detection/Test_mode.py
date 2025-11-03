import pickle

import cv2

from Face_Pose_Detection import get_face_landmarks

delay_ms = 25
emotions = ['HAPPY', 'SAD', 'SURPRISED']

with open('./model', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    face_landmarks = get_face_landmarks(frame, static_image_mode=False)

    if face_landmarks is not None:
        output = model.predict([face_landmarks])

        # Draw the predicted emotion on the frame
        cv2.putText(frame, 
                    emotions[int(output[0])], 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    3, 
                    (0, 255, 0), 
                    3)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(delay_ms) & 0xFF 
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()