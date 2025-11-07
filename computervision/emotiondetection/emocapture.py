import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time


MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media.tflite'
EMOTION_LABELS = ['loving', 'boring']
CONFIDENCE_THRESHOLD = 0.50 
INPUT_SHAPE = None 

def load_resources():
    global INPUT_SHAPE
    
    print("Loading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading TFLite model at {MODEL_PATH}: {e}")
        exit()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    INPUT_SHAPE = input_details[0]['shape'] 

    print("Loading Haar Cascade classifier...")
    HAAR_CASCADE_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if face_cascade.empty():
        print(f"Error loading Haar Cascade file at: {HAAR_CASCADE_FILE}")
        exit()
        
    print("Model and Cascade loaded successfully.")
    return interpreter, face_cascade, input_details, output_details

def initialize_picam2():
    print("Setting up Picamera2...")
    try:
        picam2 = Picamera2()
        # Ensure the main configuration matches the frame size used in the original loop
        config = picam2.create_preview_configuration(main={"size": (640, 480)}) 
        picam2.configure(config)
        picam2.start()
        # Small delay to allow the camera to warm up
        time.sleep(1) 
        print("Camera started.")
        return picam2
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        exit()

def process_frame(frame, interpreter, face_cascade, input_details, output_details):
    if INPUT_SHAPE is None:
        return ["Error: Model input shape not defined."]

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # This call already uses keyword arguments and is correct:
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    predictions_list = []

    for (x, y, w, h) in faces:
       
        roi_gray = gray[y:y + h, x:x + w]
        
        resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
        input_data = resized_face.astype('float32') / 255.0
        

        input_data = np.expand_dims(input_data, axis=0) 
        input_data = np.expand_dims(input_data, axis=-1)
        
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
    
    return predictions_list

def main():
    print("\nStarting real-time detection. Press 'q' in the window to exit.")

    try:
        while True:
            frame = picam2.capture_array() 

            # 3. Process Frame and Get Results
            emotion_results = process_frame(
                frame, 
                interpreter, 
                face_cascade, 
                input_details, 
                output_details
            )

            frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for i, result_text in enumerate(emotion_results):
                if i == 0:
                    print(f"Detected Emotion: {result_text}")
 
                gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
                # --- FIX APPLIED HERE ---
                # Added '0' for the 'flags' argument to prevent 'minSize' (30, 30) 
                # from being misinterpreted as 'flags'.
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (30, 30))
                # ------------------------
                
                if i < len(faces):
                    (x, y, w, h) = faces[i]
                    print(result_text)
                    
            print("No faces detected")

    except Exception as e:
        print(f"An unexpected error occurred during the loop: {e}")

    finally:
        print("\nCleaning up resources...")
        picam2.stop() 
        print("Program terminated.")

def run_emotion_detector():
    try:
    
        frame = picam2.capture_array() 
        
        emotion_results = process_frame(
            frame, 
            interpreter, 
            face_cascade, 
            input_details, 
            output_details
        )

        frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for i, result_text in enumerate(emotion_results):
            if i == 0:
                print(f"Detected Emotion: {result_text}")

            gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            # --- FIX APPLIED HERE ---
            # Added '0' for the 'flags' argument.
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (30, 30))
            # ------------------------
            
            if i < len(faces):
                (x, y, w, h) = faces[i]          
                return result_text.split(':')[0].strip().lower()
                  
        return None
         

    except Exception as e:
        print(f"An unexpected error occurred during the loop: {e}")




interpreter, face_cascade, input_details, output_details = load_resources()
picam2 = initialize_picam2()

if __name__ == '__main__':
    main()