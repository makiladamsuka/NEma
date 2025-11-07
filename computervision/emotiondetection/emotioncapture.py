import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
# from libcamera import controls # Not needed since we removed AfMode control

# --- Configuration ---
MODEL_PATH = 'media.tflite'
EMOTION_LABELS = ['Happy', 'Smile']
CONFIDENCE_THRESHOLD = 0.50 

# --- Model Initialization ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    # Note: If this fails, ensure 'media.tflite' is in the same directory.
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'] 

# --- Face Detector Initialization (Revised for robustness) ---
# Use the built-in path to the Haar Cascade file provided by the cv2 package
HAAR_CASCADE_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
if face_cascade.empty():
    print(f"Error loading Haar Cascade file at: {HAAR_CASCADE_FILE}")
    exit()

# --- Picamera2 Setup (Fixed) ---
print("Setting up Picamera2...")
try:
    picam2 = Picamera2()
    
    # Configure the camera for a fast preview stream
    config = picam2.create_preview_configuration(main={"size": (640, 480)}) 
    picam2.configure(config)
    
    # FIX: The AfMode control has been removed as it causes errors on fixed-focus cameras (like IMX219)
    # picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) 
    
    picam2.start()
except Exception as e:
    print(f"Error initializing Picamera2: {e}")
    exit()

print("Camera started. Press 'q' to exit.")

# --- Main Loop ---
while True:
    # Get a frame from the camera
    # capture_array() is the most efficient way to get a numpy array
    frame = picam2.capture_array() 
    
    # Convert from RGB (Picamera2 default) to BGR (OpenCV default)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- Face Detection ---
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # --- Emotion Prediction ---
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        
        # Resize and preprocess the face ROI for the TFLite model
        resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
        input_data = resized_face.astype('float32') / 255.0
        
        # Adjust array dimensions to match TFLite input (e.g., [1, H, W, 1])
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = input_data.reshape(INPUT_SHAPE) 
        
        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], input_data) 
        interpreter.invoke() 

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        max_index = np.argmax(predictions)
        max_confidence = predictions[max_index]
        
        # Determine emotion and confidence text
        if max_confidence >= CONFIDENCE_THRESHOLD:
            predicted_emotion = EMOTION_LABELS[max_index]
            confidence_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
            color = (0, 255, 0) # Green for confident prediction
        else:
            confidence_text = f"Uncertain: {max_confidence*100:.1f}%"
            color = (0, 165, 255) # Orange for uncertain prediction

        cv2.putText(frame, confidence_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # --- Display ---
    cv2.imshow('Emotion Detection', frame)

    # --- Exit condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
picam2.stop() # Stop the camera stream
cv2.destroyAllWindows()
print("Program terminated.")