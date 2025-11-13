import cv2
import numpy as np
import tensorflow as tf
import os
import time 
from picamera2 import Picamera2 

# --- Configuration ---
MODEL_PATH = 'media2.tflite'
YUNET_MODEL_PATH = 'face_detection_yunet_2023mar.onnx' 
YUNET_INPUT_SIZE = (320, 320) 

EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.50

# --- 1. Load the Models ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite emotion model: {e}") 
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'] 

try:
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
    # Confidence threshold of 0.7
=======
    # *** MODIFIED: Lowered confidence from 0.9 to 0.7 ***
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, 
        "", 
        YUNET_INPUT_SIZE, 
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
        0.7, # Good confidence value
=======
        0.7, # Lowered threshold to improve detection
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
        0.3, 
        5000 
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}") 
    exit()


# --- 2. Video Capture and Inference Loop ---

print("Initializing PiCamera2...")
picam2 = Picamera2()

<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
# *** MODIFIED: Explicitly request 4-channel XRGB8888 format ***
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "XRGB8888"}
=======
# *** MODIFIED: Request standard RGB888 format ***
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
)
picam2.configure(config)

picam2.start()
print("Camera started. Allowing 1 second for warmup...")
time.sleep(1.0) 

print("Starting detection loop...")

while True:
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
    # Capture a frame from Picamera2 (it's in 4-channel RGBA format)
    frame = picam2.capture_array()
    
    # *** NEW FIX: Convert the 4-channel RGBA frame to 3-channel BGR ***
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Now 'frame' is in the 3-channel BGR format that all cv2 functions expect.
=======
    # Capture a frame from Picamera2 (it's in RGB format)
    frame = picam2.capture_array()
    
    # *** NEW FIX: Convert the RGB frame to BGR for OpenCV ***
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Now 'frame' is in the BGR format that all cv2 functions expect.
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
    
    frame_height, frame_width, _ = frame.shape
    face_detector.setInputSize((frame_width, frame_height)) 
    
    # --- YuNet Face Detection ---
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
=======
    # We pass the BGR frame to the detector
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
    success, faces = face_detector.detect(frame)
    
    if faces is not None:
        for face_data in faces:
            x, y, w, h = map(int, face_data[:4]) 
            
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
=======
            # Draw rectangle on the BGR frame
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            print(f"facecoods {(x, y)},  {(x+w, y+h)}")
            
            x_end = min(x + w, frame_width)
            y_end = min(y + h, frame_height)
            x_start = max(0, x)
            y_start = max(0, y)

            # Crop from the BGR frame
            roi_color = frame[y_start:y_end, x_start:x_end]
            
            if roi_color.size == 0:
                continue 
            
            # Convert the BGR crop to Grayscale
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
=======
            # This line is now correct, as it expects a BGR input
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # --- Pre-process the face for the emotion model ---
            resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
            input_data = resized_face.astype('float32') / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.expand_dims(input_data, axis=-1)
            
            if input_data.shape != tuple(INPUT_SHAPE):
                input_data = input_data.reshape(INPUT_SHAPE) 

            # --- Run TFLite Inference ---
            interpreter.set_tensor(input_details[0]['index'], input_data) 
            interpreter.invoke()

            predictions = interpreter.get_tensor(output_details[0]['index'])
            max_index = np.argmax(predictions[0])
            predicted_emotion = EMOTION_LABELS[max_index]
            max_confidence = predictions[0][max_index] 

            # --- Confidence Check and Text Display ---
            if max_confidence >= CONFIDENCE_THRESHOLD:
                confidence_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
                color = (0, 255, 0) # Green
            else:
                confidence_text = "Uncertain" 
                color = (255, 255, 255) # White
                
<<<<<<< Updated upstream:computervision/emotiondetection/emotioncapturev2.py
=======
            # Display text on the BGR frame
>>>>>>> Stashed changes:computervision/emotiondetection/emotiondetectionpi.py
            cv2.putText(frame, confidence_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the final (correct color) BGR frame
    cv2.imshow('Emotion Detection (PiCamera2 + YuNet)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. Cleanup ---
print("Stopping camera...")
picam2.stop() 
cv2.destroyAllWindows()