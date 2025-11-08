import cv2
import numpy as np
import tensorflow as tf
import os
import time  # *** MODIFIED: Added for camera warm-up ***
# *** MODIFIED: Import the Picamera2 library ***
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
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, 
        "", 
        YUNET_INPUT_SIZE, 
        0.9, 
        0.3, 
        5000 
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}") 
    exit()


# --- 2. Video Capture and Inference Loop ---

# *** MODIFIED: Initialize Picamera2 ***
print("Initializing PiCamera2...")
picam2 = Picamera2()

# Configure the camera. We ask for a BGR888 format so OpenCV can use it directly.
# A 640x480 resolution is fast and good for detection.
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"}
)
picam2.configure(config)

# Start the camera stream
picam2.start()
print("Camera started. Allowing 1 second for warmup...")
# Give the camera a moment to warm up
time.sleep(1.0) 

print("Starting detection loop...")

while True:
    # *** MODIFIED: Capture a frame from Picamera2 ***
    # This directly gives us a NumPy array in the BGR format we asked for.
    frame = picam2.capture_array()
    
    # The 'ret' check is no longer needed, as picam2 will 
    # raise an error if capture fails.
    
    # Get the current frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Set the input size for YuNet to the current frame size. 
    face_detector.setInputSize((frame_width, frame_height)) 
    
    # --- YuNet Face Detection (Replaces Haar Cascade) ---
    success, faces = face_detector.detect(frame)
    
    if faces is not None:
        for face_data in faces:
            x, y, w, h = map(int, face_data[:4]) 
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop the face region (color)
            x_end = min(x + w, frame_width)
            y_end = min(y + h, frame_height)
            x_start = max(0, x)
            y_start = max(0, y)

            roi_color = frame[y_start:y_end, x_start:x_end]
            
            # 💡 CRITICAL FIX: Check if the ROI is empty
            if roi_color.size == 0:
                continue 
            
            # Convert the cropped face to grayscale
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
                
            # Display the result on the frame
            cv2.putText(frame, confidence_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the final frame
    cv2.imshow('Emotion Detection (PiCamera2 + YuNet)', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. Cleanup ---
print("Stopping camera...")
# *** MODIFIED: Stop the Picamera2 stream ***
picam2.stop() 
cv2.destroyAllWindows()