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

# Assuming oled.emodisplay is in a location Python can find
from oled.emodisplay import setup_and_start_display, display_emotion

# --- 1. Setup ---
setup_and_start_display()
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# Servo Hardware Setup
I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90

# --- MODIFIED: Added TILT_DOWN_ANGLE ---
TILT_DOWN_ANGLE = 125 # Angle to look down when sad (e.g., 120 degrees)

# PID and Smoothing
PAN_Kp, PAN_Ki, PAN_Kd =  4/10, .001, 9/10
TILT_Kp, TILT_Ki, TILT_Kd = 4/10, .001, 9/10

SMOOTHING_FACTOR = .008
RETURN_SMOOTHING_FACTOR = 1 # --- Faster smoothing for non-tracking moves
PID_MAX_OFFSET = 60

# Model Paths
MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media2.tflite'
YUNET_MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/face_detection_yunet_2023mar.onnx'
YUNET_INPUT_SIZE = (320, 320)
EMOTION_LABELS = ['Happy','Smile'] # Make sure this matches your model
CONFIDENCE_THRESHOLD = 0.50

# --- State Variables ---
last_face_x = FRAME_WIDTH / 2
last_face_y = FRAME_HEIGHT / 2
IS_SEARCHING = False
search_frame_counter = 0

MAX_SEARCH_FRAMES = 50 # How long to search at last position
SAD_TILT_FRAMES = 150  # How many frames to stay in the "sad tilt" phase
can_be_sad = False     # Flag to allow sad state only after seeing a face

# --- 2. Model Initialization (TFLite Emotion) ---
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

# --- 3. Model Initialization (YuNet Face) ---
try:
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        YUNET_INPUT_SIZE,
        0.4, # Confidence threshold
        0.3, # NMS threshold
        5000
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}")
    sys.exit(1)

# --- 4. Hardware Initialization (Servos) ---
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

    print(f"PCA9685 initialized. Servos set to {PAN_CENTER}°, {TILT_CENTER}°.")

except Exception as e:
    print(f"Error initializing hardware: {e}")
    sys.exit(1)

# --- 5. PID and Camera Setup ---
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
# --- 6. MAIN CONTROL LOOP ---
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
        
        # Convert RGB888 (from PiCamera) to BGR (for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        face_detector.setInputSize((FRAME_WIDTH, FRAME_HEIGHT))
        
        success, faces = face_detector.detect(frame)

        pan_offset = 0
        tilt_offset = 0
        emotion_text = "Searching..."
        emotion_color = (255, 255, 255)
        
        # --- NEW: Flag for smoothing speed ---
        use_fast_smoothing = False

        # --- TRACKING LOGIC ---
        if faces is not None:
            IS_SEARCHING = False # Found the face, stop searching!
            can_be_sad = True    # We've seen a face, so we *can* be sad later
            search_frame_counter = 0
            use_fast_smoothing = False # Use slow, smooth tracking

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
            
            # --- Emotion Detection ---
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
                    # display_emotion("neutral") # Or some other default
            
            # Drawing for Face and Emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
            
        else: # NO FACE DETECTED
            IS_SEARCHING = True
            display_emotion("idle")
            
            # --- Always increment the counter when a face is not seen ---
            search_frame_counter += 1

            # --- Phase 1: Searching at Last Known Position ---
            if search_frame_counter < MAX_SEARCH_FRAMES:
                pan_offset = pan_pid(last_face_x)
                tilt_offset = tilt_pid(last_face_y)
                emotion_text = f"Searching ({search_frame_counter}/{MAX_SEARCH_FRAMES})"
                emotion_color = (255, 0, 255) # Magenta
                use_fast_smoothing = False # Use slow searching
                display_emotion("looking")
                
                # Draw a target on the Last Known Position
                cv2.circle(frame, (int(last_face_x), int(last_face_y)), 10, (255, 0, 255), 2)
                cv2.putText(frame, "LAST POS", (int(last_face_x) + 15, int(last_face_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # --- MODIFIED: Phase 2: Sad Tilt Down ---
            # This 'elif' block will run as long as the counter is in range
            # AND the sad emotion has been "armed" (can_be_sad == True)
            elif search_frame_counter < (MAX_SEARCH_FRAMES + SAD_TILT_FRAMES) and can_be_sad:
                if can_be_sad :
                    display_emotion("sad")
                    can_be_sad = False
                    
                
                
                emotion_color = (255, 100, 0) # Blue/Cyan for sad
                
                pan_offset = 0 # Look straight ahead
                
                # This calculation will make the target angle TILT_DOWN_ANGLE
                tilt_offset = TILT_CENTER - TILT_DOWN_ANGLE
                
                use_fast_smoothing = True # Use fast smoothing to tilt down

            # --- Phase 3: Give up and return to center ---
            # This 'else' block triggers when:
            # 1. The sad animation time is over (counter > 200)
            # 2. OR the sad emotion was never armed (can_be_sad == False)
            else:
                pan_offset = 0
                tilt_offset = 0
                emotion_text = "Idle"
                emotion_color = (128, 128, 128) # Gray
                
                # --- THIS IS THE FIX ---
                # We disarm the 'sad' flag HERE.
                # It will now stay in this idle state until a face is seen again.
                can_be_sad = False 
                
                display_emotion("Idle")
                use_fast_smoothing = True # Use fast smoothing to return to center
        
        # --- Servo Angle Calculation ---
        target_pan_angle = PAN_CENTER + pan_offset
        target_tilt_angle = TILT_CENTER - tilt_offset # Tilt is inverted

        # --- NEW: Conditional Smoothing Logic ---
        if use_fast_smoothing:
            # GO TO TARGET FAST: Use the faster return smoothing factor
            current_pan_angle = (target_pan_angle * RETURN_SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - RETURN_SMOOTHING_FACTOR))
            current_tilt_angle = (target_tilt_angle * RETURN_SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - RETURN_SMOOTHING_FACTOR))
        else:
            # APPLY NORMAL SMOOTHING: We are tracking or searching slowly
            current_pan_angle = (target_pan_angle * SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - SMOOTHING_FACTOR))
            current_tilt_angle = (target_tilt_angle * SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - SMOOTHING_FACTOR))
        
        # Clamping (always apply this)
        current_pan_angle = max(0, min(180, current_pan_angle))
        current_tilt_angle = max(0, min(180, current_tilt_angle))

        # Move the Servos
        pan_servo.angle = current_pan_angle
        tilt_servo.angle = current_tilt_angle
        
        
        # --- Display and OpenCV Window ---
        # Draw Setpoint
        cv2.circle(frame, (int(PAN_SETPOINT), int(TILT_SETPOINT)), 5, (0, 0, 255), -1)
        # Draw current emotion/state text
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_color, 2)
        
        cv2.imshow('YuNet Face Tracking & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    cleanup_and_exit()