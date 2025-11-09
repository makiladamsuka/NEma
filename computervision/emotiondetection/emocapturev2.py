import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time
import os

# --- Configuration ---
MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media2.tflite'
YUNET_MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/face_detection_yunet_2023mar.onnx' 
# YuNet uses a dynamic input size, but this is its internal size for initialization
YUNET_INPUT_SIZE = (320, 320) 
FRAME_WIDTH, FRAME_HEIGHT = 640, 480 

EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.50 
INPUT_SHAPE = None # Will be set during resource loading

# --- Resource Loading Functions ---
def load_resources():
    """Loads the TFLite emotion model and the YuNet face detector."""
    global INPUT_SHAPE
    
    print("Loading TFLite emotion model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading TFLite model at {MODEL_PATH}: {e}")
        exit()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    INPUT_SHAPE = input_details[0]['shape'] 

    print("Loading YuNet Face Detector...")
    try:
        # YuNet is more robust than Haar Cascade
        face_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, 
            "", 
            YUNET_INPUT_SIZE, 
            0.7, # Confidence threshold
            0.3, 
            5000 
        )
        if face_detector is None:
             raise RuntimeError("cv2.FaceDetectorYN.create returned None")
    except Exception as e:
        print(f"Error loading YuNet model at {YUNET_MODEL_PATH}: {e}") 
        exit()
        
    print("Models loaded successfully.")
    return interpreter, face_detector, input_details, output_details

def initialize_picam2():
    """Sets up and starts the Picamera2 instance."""
    print("Setting up Picamera2...")
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"} 
        )
        picam2.configure(config)
        picam2.start()
        # Small delay to allow the camera to warm up
        time.sleep(1) 
        print("Camera started.")
        return picam2
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        exit()

# --- Core Processing Function ---
def process_frame(frame, interpreter, face_detector, input_details, output_details):
    """
    Performs face detection and emotion prediction on a single frame.
    
    Returns:
        tuple: (list of emotion/confidence strings, list of face coordinates)
    """
    if INPUT_SHAPE is None:
        return [], [] # Return empty list on error
    
    frame_height, frame_width, _ = frame.shape
    face_detector.setInputSize((frame_width, frame_height)) 
    
    # 1. Face Detection (YuNet is already faster and more accurate)
    success, faces = face_detector.detect(frame)
    
    predictions_list = []
    filtered_faces = [] 

    # 2. Emotion Prediction
    if faces is not None:
        for face_data in faces:
            # face_data format: [bbox_x, bbox_y, bbox_w, bbox_h, confidence, 5_landmarks...]
            x, y, w, h = map(int, face_data[:4]) 
            
            # No need for MIN_FACE_SIZE filter here, as YuNet's confidence threshold (0.7) 
            # and minFaceSize (which is handled internally or via the min_overlap parameter in the API) 
            # is generally better than a fixed pixel size.
            
            x_end = min(x + w, frame_width)
            y_end = min(y + h, frame_height)
            x_start = max(0, x)
            y_start = max(0, y)

            roi_color = frame[y_start:y_end, x_start:x_end]
            
            if roi_color.size == 0:
                continue 
            
            # Save the face location for drawing later
            filtered_faces.append((x, y, w, h))

            # Convert the BGR crop to Grayscale for the emotion model
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # Preprocessing
            resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
            input_data = resized_face.astype('float32') / 255.0
            
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.expand_dims(input_data, axis=-1)
            
            # Check shape compatibility just in case
            if input_data.shape != tuple(INPUT_SHAPE):
                input_data = input_data.reshape(INPUT_SHAPE) 

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data) 
            interpreter.invoke() 

            # Get results
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            max_index = np.argmax(predictions)
            max_confidence = predictions[max_index]
            
            # Format the result string
            if max_confidence >= CONFIDENCE_THRESHOLD:
                predicted_emotion = EMOTION_LABELS[max_index]
                confidence_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
            else:
                confidence_text = f"Uncertain: {max_confidence*100:.1f}% (Predicted: {EMOTION_LABELS[max_index]})"

            predictions_list.append(confidence_text)
            
    return predictions_list, filtered_faces # Return both results and face locations

# --- Main Functions (Updated for Display) ---
def main(picam2, interpreter, face_detector, input_details, output_details):
    """Main loop for capturing frames and displaying results."""
    print("\nStarting real-time detection. Press 'q' in the window to exit.")

    try:
        while True:
            # Capture a frame (it's in 4-channel XRGB8888 format)
            frame = picam2.capture_array() 
            
            # Convert the 4-channel XRGB frame to 3-channel BGR
            # This is essential as cv2 functions generally expect BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # 1. Process Frame and Get Results and Faces
            emotion_results, faces = process_frame(
                frame_bgr, 
                interpreter, 
                face_detector, 
                input_details, 
                output_details
            )

            # 2. Draw Results and Display
            if not faces:
                cv2.putText(frame_bgr, "No Face Detected", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            for i, result_text in enumerate(emotion_results):
                (x, y, w, h) = faces[i]
                
                # Print result (only the first face is printed to avoid spam)
                if i == 0:
                    print(f"Detected Emotion: {result_text}")
                
                # Draw bounding box
                color = (0, 255, 0) if "Uncertain" not in result_text else (0, 165, 255)
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
                
                # Draw text
                cv2.putText(frame_bgr, result_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Display the frame
            cv2.imshow('Emotion Detection (YuNet)', frame_bgr)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An unexpected error occurred during the loop: {e}")

    finally:
        print("\nCleaning up resources...")
        picam2.stop() 
        cv2.destroyAllWindows()
        print("Program terminated.")


interpreter, face_detector, input_details, output_details = load_resources()
picam2 = initialize_picam2()

# The run_emotion_detector function is simplified for a single-frame use case
def run_emotion_detector(picam2=picam2, interpreter=interpreter, face_detector=face_detector, input_details=input_details, output_details=output_details):
    """Captures a single frame, processes it, and returns the top emotion."""
    try:
        # Get a single frame
        frame = picam2.capture_array() 
        
        # Convert the 4-channel XRGB frame to 3-channel BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Process frame
        emotion_results, faces = process_frame(
            frame_bgr, 
            interpreter, 
            face_detector, 
            input_details, 
            output_details
        )
        
        if emotion_results:
            # The emotion is the part before the colon
            emotion = emotion_results[0].split(':')[0].strip().lower()
            face_coordinates = faces[0] # (x, y, w, h)
            return emotion, face_coordinates
            
        return None, None # No face found
            
    except Exception as e:
        print(f"An unexpected error occurred in run_emotion_detector: {e}")
        return None, None




if __name__ == '__main__':
    main(picam2, interpreter, face_detector, input_details, output_details)