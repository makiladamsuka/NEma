import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

MODEL_FILENAME = 'media.h5'
IMAGE_SIZE = (48, 48)

EMOTION_LABELS = ['Angry','Disgust','Fear','Happy', 'Sad','Natural', 'Suprise']
FONT = cv2.FONT_HERSHEY_SIMPLEX
PREDICTION_THRESHOLD = 0.4

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists(MODEL_FILENAME):
    exit()
    
try:
    model = load_model(MODEL_FILENAME)
except Exception as e:
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7) as face_detection:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape

                x = int(bbox_c.xmin * iw)
                y = int(bbox_c.ymin * ih)
                w = int(bbox_c.width * iw)
                h = int(bbox_c.height * ih)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi_gray = frame[y:y + h, x:x + w]
                
                if roi_gray.size != 0:

                    roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    roi_array = img_to_array(roi_resized)
                    roi_array = roi_array / 255.0
                    roi_array = np.expand_dims(roi_array, axis=0)
                    roi_array = np.expand_dims(roi_array, axis=-1)

                    predictions = model.predict(roi_array, verbose=0)[0]

                    predicted_index = np.argmax(predictions)
                    confidence = predictions[predicted_index]

                    if confidence > PREDICTION_THRESHOLD:
                        predicted_emotion = EMOTION_LABELS[predicted_index]
                        label = f"{predicted_emotion}: {confidence*100:.1f}%"
                    else:
                        label = "Detecting..."

                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        FONT,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

        cv2.imshow('Real-Time Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
