#improved returrn home


import cv2
import numpy as np
import tensorflow as tf
import os
import time
from picamera2 import Picamera2
from simple_pid import PID
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import sys 

from oled.emodisplay import setup_and_start_display, display_emotion


setup_and_start_display() 
FRAME_WIDTH, FRAME_HEIGHT = 640, 480 

# Servo Hardware Setup
I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90


PAN_Kp, PAN_Ki, PAN_Kd =  4/10, .001, 9/10
TILT_Kp, TILT_Ki, TILT_Kd = 4/10, .001, 9/10

SMOOTHING_FACTOR = .008
RETURN_SMOOTHING_FACTOR = 0.09 
PID_MAX_OFFSET = 60

# Model Paths (Your provided paths)
MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media2.tflite'
YUNET_MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/face_detection_yunet_2023mar.onnx' 
YUNET_INPUT_SIZE = (320, 320) 
EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.50


# --- NEW: Persistence Variables ---
# Initialize the last known center of the face to the center of the frame
last_face_x = FRAME_WIDTH / 2
last_face_y = FRAME_HEIGHT / 2
# Set a flag to easily check if we should be searching
IS_SEARCHING = False 
# How many times to use the last position before giving up and going to center
MAX_SEARCH_FRAMES = 50
search_frame_counter = 0



try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite emotion model: {e}") 
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'] 
TFLITE_INPUT_H, TFLITE_INPUT_W = INPUT_SHAPE[1], INPUT_SHAPE[2]

try:
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, 
        "", 
        YUNET_INPUT_SIZE, 
        0.4, # Your set confidence
        0.3, 
        5000 
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}") 
    sys.exit(1)


try:
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50 

    pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
    tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    current_pan_angle = PAN_CENTER
    current_tilt_angle = TILT_CENTER

    print(f"PCA9685 initialized. Servos set to {PAN_CENTER}°, {TILT_CENTER}° tilt.")

except ValueError:
    print("Error: Could not find PCA9685 at the specified I2C address.")
    sys.exit(1)
except ImportError as e:
    print(f"Error: Required library not found ({e}). Ensure adafruit-blinka and adafruit-circuitpython-pca9685 are installed.")
    sys.exit(1)


PAN_SETPOINT = FRAME_WIDTH / 2
TILT_SETPOINT = FRAME_HEIGHT / 2

pan_pid = PID(PAN_Kp, PAN_Ki, PAN_Kd, setpoint=PAN_SETPOINT)
pan_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)

tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
tilt_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)


picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"} 
)

picam2.configure(config)
picam2.start()
time.sleep(1.0) 
print(f"Picamera2 started at {FRAME_WIDTH}x{FRAME_HEIGHT}.")


# =================================================================
# --- 5. MAIN CONTROL LOOP ---
# =================================================================

def cleanup_and_exit():
    """Stops the camera, resets servos, and closes OpenCV windows."""
    print("\nStopping camera and resetting servos...")
    picam2.stop()
    cv2.destroyAllWindows()
    pan_servo.angle = PAN_CENTER 
    tilt_servo.angle = TILT_CENTER
    time.sleep(0.5) 
    sys.exit(0)

try:
    while True:
        frame = picam2.capture_array()
        
        # *** IMPORTANT FIX: Convert RGB888 to BGR for OpenCV functions (e.g., drawing) ***
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        face_detector.setInputSize((FRAME_WIDTH, FRAME_HEIGHT)) 
        
        success, faces = face_detector.detect(frame)

        target_pan_angle = PAN_CENTER
        target_tilt_angle = TILT_CENTER
        emotion_text = "Searching..."
        emotion_color = (255, 255, 255)
        
        # --- TRACKING LOGIC ---
        if faces is not None:
            IS_SEARCHING = False # Found the face, stop searching!
            search_frame_counter = 0

            # Get face coordinates and center
            (x, y, w, h) = map(int, faces[0][:4]) 
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # --- Update Last Known Position ---
            last_face_x = face_center_x
            last_face_y = face_center_y

            # PID Input is the current face center
            pan_offset = pan_pid(face_center_x)
            tilt_offset = tilt_pid(face_center_y)
            
            # --- Emotion Detection (As before) ---
            x_end = min(x + w, FRAME_WIDTH)
            y_end = min(y + h, FRAME_HEIGHT)
            x_start = max(0, x)
            y_start = max(0, y)

            roi_color = frame[y_start:y_end, x_start:x_end]
            
            if roi_color.size > 0:
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(roi_gray, (TFLITE_INPUT_W, TFLITE_INPUT_H), interpolation=cv2.INTER_AREA)
                input_data = resized_face.astype('float32') / 255.0
                input_data = np.expand_dims(input_data, axis=0)
                input_data = np.expand_dims(input_data, axis=-1)
                
                if input_data.shape != tuple(INPUT_SHAPE):
                    input_data = input_data.reshape(INPUT_SHAPE) 

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                max_index = np.argmax(predictions[0])
                max_confidence = predictions[0][max_index]
                
                if max_confidence >= CONFIDENCE_THRESHOLD:
                    predicted_emotion = EMOTION_LABELS[max_index]
                    emotion_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
                    emotion_color = (0, 255, 0)
                    display_emotion(predicted_emotion) 
                else:
                    emotion_text = "Tracking..."
                    emotion_color = (255, 255, 0) # Yellow for tracking
            
            # Drawing for Face and Emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
            
        else: # NO FACE DETECTED
            IS_SEARCHING = True
            if search_frame_counter < MAX_SEARCH_FRAMES:
                # Use the LAST KNOWN position as the PID input (Momentum)
                pan_offset = pan_pid(last_face_x)
                tilt_offset = tilt_pid(last_face_y)
                emotion_text = f"Searching ({search_frame_counter}/{MAX_SEARCH_FRAMES})"
                emotion_color = (255, 0, 255) # Magenta for searching
                search_frame_counter += 1
                
                # Draw a target on the Last Known Position to visualize the search
                cv2.circle(frame, (int(last_face_x), int(last_face_y)), 10, (255, 0, 255), 2)
                cv2.putText(frame, "LAST POS", (int(last_face_x) + 15, int(last_face_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            else:
                display_emotion("sad") 
                pan_offset = 0
                tilt_offset = 0
                emotion_text = "Idle"
                emotion_color = (128, 128, 128) # Gray
                pan_pid.reset()
                tilt_pid.reset()
                

    
    
    
        
        
        # --- Servo Angle Calculation ---
        target_pan_angle = PAN_CENTER + pan_offset  
        target_tilt_angle = TILT_CENTER - tilt_offset  

        # --- NEW: Conditional Smoothing ---
        # Check if we are in the "give up and go to center" state
        is_returning_to_center = (IS_SEARCHING and search_frame_counter >= MAX_SEARCH_FRAMES)

        if is_returning_to_center:
            # GO TO CENTER FAST: Use the new, faster return smoothing factor
            current_pan_angle = (target_pan_angle * RETURN_SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - RETURN_SMOOTHING_FACTOR))
            current_tilt_angle = (target_tilt_angle * RETURN_SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - RETURN_SMOOTHING_FACTOR))
        else:
            # APPLY NORMAL SMOOTHING: We are tracking or searching at the last-known spot
            current_pan_angle = (target_pan_angle * SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - SMOOTHING_FACTOR))
            current_tilt_angle = (target_tilt_angle * SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - SMOOTHING_FACTOR))
        
        # Clamping (always apply this)
        current_pan_angle = max(0, min(180, current_pan_angle))
        current_tilt_angle = max(0, min(180, current_tilt_angle))

        # Move the Servos
        pan_servo.angle = current_pan_angle
        tilt_servo.angle = current_tilt_angle
        
        
        
        
        
        
        # Draw Setpoint
        cv2.circle(frame, (int(PAN_SETPOINT), int(TILT_SETPOINT)), 5, (0, 0, 255), -1)
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_color, 2)
        
        cv2.imshow('YuNet Face Tracking & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    cleanup_and_exit()