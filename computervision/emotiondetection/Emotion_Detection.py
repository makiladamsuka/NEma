import cv2
import numpy as np
import tensorflow as tf
import os

# --- Configuration ---
# You need to replace these with your actual file paths
MODEL_PATH = 'media2.tflite'
# **NEW: YuNet model path and configuration**
YUNET_MODEL_PATH = 'face_detection_yunet_2023mar.onnx' 
# YuNet input size (recommended for a good balance of speed and accuracy)
YUNET_INPUT_SIZE = (320, 320) 

# List of emotions corresponding to the model's output indices
EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.50

# --- 1. Load the Models ---
try:
    # Load the TFLite emotion model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    # Print the error for better debugging
    print(f"Error loading TFLite emotion model: {e}") 
    exit()

# Get input and output tensor details for the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Expected input shape from the TFLite model (e.g., [1, 48, 48, 1])
INPUT_SHAPE = input_details[0]['shape'] 

# **NEW: Load the YuNet Face Detector**
try:
    # Create the YuNet detector instance
    # Confidence threshold: 0.9 is strict; lower it (e.g., 0.6) if faces are missed.
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, 
        "", # Config file (optional)
        YUNET_INPUT_SIZE, 
        0.9, # Confidence threshold
        0.3, # NMS threshold
        5000 # Top K
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}") 
    exit()


# --- 2. Video Capture and Inference Loop ---
cap = cv2.VideoCapture(0) # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the current frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Set the input size for YuNet to the current frame size. 
    # This is crucial for YuNet to output correct pixel coordinates.
    face_detector.setInputSize((frame_width, frame_height)) 
    
    # --- YuNet Face Detection (Replaces Haar Cascade) ---
    # YuNet returns a tuple: (success, faces)
    # faces is a NumPy array where each row is [x, y, w, h, ...]
    success, faces = face_detector.detect(frame)
    
    # Check if any faces were detected
    if faces is not None:
        # faces is an array of detected faces data
        for face_data in faces:
            # YuNet bounding box is the first 4 elements: x, y, w, h
            x, y, w, h = map(int, face_data[:4]) 
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop the face region (color)
            # Ensure the crop coordinates are within the frame boundaries
            x_end = min(x + w, frame_width)
            y_end = min(y + h, frame_height)
            
            # Corrected slicing to avoid negative indices or out-of-bounds start
            x_start = max(0, x)
            y_start = max(0, y)

            roi_color = frame[y_start:y_end, x_start:x_end]
            
            # 💡 CRITICAL FIX: Check if the ROI is empty before using it
            if roi_color.size == 0:
                # This ensures cv2.cvtColor is only called on a valid image
                continue 
            
            # Convert the cropped face to grayscale for the TFLite emotion model
            # This is done only on the ROI, not the entire frame, for efficiency.
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # Pre-process the face for the emotion model
            
            # 1. Resize to the model's expected input size (e.g., 48x48)
            # Using roi_gray as input
            resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
            
            # 2. Convert to float32 and normalize
            input_data = resized_face.astype('float32') / 255.0

            # 3. Correct Reshaping: Add the Batch (0) and Channel (3) dimensions: (1, H, W, 1)
            input_data = np.expand_dims(input_data, axis=0) # Add Batch dimension
            input_data = np.expand_dims(input_data, axis=-1) # Add Channel dimension
            
            # Ensure the final shape matches the expected shape (Crucial for TFLite)
            if input_data.shape != tuple(INPUT_SHAPE):
                 input_data = input_data.reshape(INPUT_SHAPE) 


            # --- Run TFLite Inference ---
            interpreter.set_tensor(input_details[0]['index'], input_data) 
            interpreter.invoke()

            # Get the prediction result
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            # Find the index of the highest prediction probability
            max_index = np.argmax(predictions[0])
            
            # Get the predicted emotion label (string)
            predicted_emotion = EMOTION_LABELS[max_index]
            
            # Get the numerical confidence score (float)
            max_confidence = predictions[0][max_index] 

            # --- Confidence Check and Text Display ---
            if max_confidence >= CONFIDENCE_THRESHOLD:
                # Display confidence with percentage
                confidence_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
                color = (0, 255, 0) # Green for confident prediction
            else:
                # Low confidence prediction
                confidence_text = "Uncertain" # Or "Neutral"
                color = (255, 255, 255) # White color
                

            # Display the result on the frame
            # Use original x, y for text position
            cv2.putText(frame, confidence_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the final frame
    cv2.imshow('Emotion Detection (YuNet Fast Face Detection)', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. Cleanup ---
cap.release()
cv2.destroyAllWindows()