import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time
from simple_pid import PID  
from adafruit_servokit import ServoKit 

# --- 1. CONFIGURATION & TUNING ---
# File Paths (Change these to match your system!)
MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media.tflite'
HAAR_CASCADE_FILE = '/home/nema/Documents/NEma/computervision/emotiondetection/haarcascade_frontalface_default.xml'

# Camera/Display Settings
WIDTH, HEIGHT = 1920, 1080  # Full HD resolution from your first script
FRAME_CENTER_X, FRAME_CENTER_Y = WIDTH // 2, HEIGHT // 2
MIN_FACE_SIZE = 80  
EMOTION_LABELS = ['loving', 'boring']
CONFIDENCE_THRESHOLD = 0.50 

# PID GAIN TUNING (Adjust these for smooth performance!)
PID_PAN_GAINS = (0.012, 0.0001, 0.0005)  # (Kp, Ki, Kd) - Pan (Left/Right)
PID_TILT_GAINS = (0.010, 0.0001, 0.0004) # Tilt (Up/Down)

# Servo Limits (in degrees)
PAN_MIN = 30  
PAN_MAX = 150 
TILT_MIN = 60 
TILT_MAX = 120 

# Global objects
interpreter, face_cascade, input_details, output_details = None, None, None, None
picam2 = None
servo_kit = None
current_pan_angle = 90
current_tilt_angle = 90
INPUT_SHAPE = None

# --- 2. RESOURCE LOADING FUNCTIONS ---

def load_resources():
    """Loads the TFLite model and Haar Cascade classifier."""
    global INPUT_SHAPE, interpreter, face_cascade, input_details, output_details
    print("Loading resources...")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading TFLite model: {e}"); exit()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    INPUT_SHAPE = input_details[0]['shape'] 

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if face_cascade.empty():
        print(f"Error loading Haar Cascade file: {HAAR_CASCADE_FILE}"); exit()
    print("Model and Cascade loaded successfully.")

def initialize_picam2():
    """Sets up and starts the Picamera2."""
    global picam2
    print("Setting up Picamera2...")
    try:
        picam2 = Picamera2()
        # Use the high-resolution setting
        config = picam2.create_preview_configuration(main={"size": (WIDTH, HEIGHT)})  
        picam2.configure(config)
        picam2.start()
        time.sleep(1) 
        print("Camera started.")
    except Exception as e:
        print(f"Error initializing Picamera2: {e}"); exit()

def initialize_pca9685():
    """Initializes the PCA9685 servo driver and centers servos."""
    global servo_kit, current_pan_angle, current_tilt_angle
    print("Initializing PCA9685...")
    try:
        kit = ServoKit(channels=16) 
        kit.servo[0].angle = current_pan_angle # Pan Servo
        kit.servo[1].angle = current_tilt_angle # Tilt Servo
        print("PCA9685 initialized. Servos centered.")
        servo_kit = kit
        return kit
    except Exception as e:
        print(f"Error initializing PCA9685/I2C: {e}"); exit()

# --- 3. CORE PROCESSING FUNCTIONS ---

def process_frame(frame):
    """Detects faces, filters by size, and runs emotion inference on the largest face."""
    if INPUT_SHAPE is None: return None, None 

    # Convert to grayscale for Haar Cascade detection
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find the largest face that meets the minimum size requirement
    largest_face = None
    max_area = 0
    for (x, y, w, h) in faces:
        if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)

    if largest_face is None:
        return None, None # No suitable face found

    x, y, w, h = largest_face
    
    # Emotion detection (only on the largest face)
    roi_gray = gray[y:y + h, x:x + w]
    resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
    input_data = resized_face.astype('float32') / 255.0
    input_data = np.expand_dims(input_data, axis=0)  
    input_data = np.expand_dims(input_data, axis=-1)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)  
    interpreter.invoke()  
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    max_index = np.argmax(predictions)
    max_confidence = predictions[max_index]
    
    if max_confidence >= CONFIDENCE_THRESHOLD:
        predicted_emotion = EMOTION_LABELS[max_index]
        emotion_text = f"{predicted_emotion.upper()}: {max_confidence*100:.1f}%"
    else:
        emotion_text = f"UNCERTAIN: {max_confidence*100:.1f}%"
    
    return emotion_text, largest_face

def calculate_errors(face_coords):
    """Calculates pixel error from the frame center."""
    x, y, w, h = face_coords
    
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    pan_error = face_center_x - FRAME_CENTER_X
    tilt_error = face_center_y - FRAME_CENTER_Y
    
    return pan_error, tilt_error

def update_servos(kit, pan_adjustment, tilt_adjustment):
    """Applies PID adjustments to servos, respecting min/max limits."""
    global current_pan_angle, current_tilt_angle
    
    new_pan_angle = current_pan_angle + pan_adjustment 
    new_tilt_angle = current_tilt_angle + tilt_adjustment
    
    # Clip the angle to stay within safe mechanical limits
    new_pan_angle = max(PAN_MIN, min(PAN_MAX, new_pan_angle))
    new_tilt_angle = max(TILT_MIN, min(TILT_MAX, new_tilt_angle))

    # Only send commands if the change is significant enough
    if abs(new_pan_angle - current_pan_angle) > 0.1:
        kit.servo[0].angle = new_pan_angle
        current_pan_angle = new_pan_angle
        
    if abs(new_tilt_angle - current_tilt_angle) > 0.1:
        kit.servo[1].angle = new_tilt_angle
        current_tilt_angle = new_tilt_angle
        
    return new_pan_angle, new_tilt_angle

# --- 4. MAIN TRACKING LOOP ---

def face_tracking_loop(kit):
    print("\nSetting up PID controllers...")
    
    # Setpoint is 0 (zero error)
    pid_pan = PID(Kp=PID_PAN_GAINS[0], Ki=PID_PAN_GAINS[1], Kd=PID_PAN_GAINS[2], setpoint=0, sample_time=0.05, output_limits=(-5.0, 5.0))
    pid_tilt = PID(Kp=PID_TILT_GAINS[0], Ki=PID_TILT_GAINS[1], Kd=PID_TILT_GAINS[2], setpoint=0, sample_time=0.05, output_limits=(-5.0, 5.0))
    
    print("Starting PID-controlled face tracking with live feed...")
    
    while True:
        try:
            # 1. Capture the raw frame (RGB format)
            raw_frame = picam2.capture_array() 
            raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_180) # Adjust for mounting

            # Convert RGB to BGR for OpenCV display and drawing
            frame_display = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

            # 2. Process the frame for detection/emotion
            emotion_text, face_coords = process_frame(raw_frame) 
            
            # Draw the center crosshair (Target Goal)
            cv2.circle(frame_display, (FRAME_CENTER_X, FRAME_CENTER_Y), 10, (255, 0, 0), 2) # Blue circle

            if face_coords is not None:
                # 3. Calculate Error and Get PID Adjustment
                pan_error, tilt_error = calculate_errors(face_coords)
                
                # The PID controller calculates the *adjustment* needed to move the camera
                # We feed -pan_error because a positive error (face on the right) needs a negative
                # adjustment (camera moves left) to center.
                pan_adjustment = pid_pan(-pan_error) 
                tilt_adjustment = pid_tilt(tilt_error) 
                
                # 4. Apply PID output to the servos
                new_pan, new_tilt = update_servos(kit, pan_adjustment, tilt_adjustment)
                
                # 5. DRAW BOUNDING BOX AND TEXT 
                (x, y, w, h) = face_coords
                color = (0, 255, 0) # Green BGR color
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Draw box, center dot, and emotion text
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame_display, (face_center_x, face_center_y), 5, (0, 0, 255), -1) # Red dot on face
                cv2.putText(frame_display, emotion_text, (x, y - 10),  
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                print(f"Tracking: {emotion_text.split(':')[0].strip()} | Pan: {new_pan:.1f}° | Tilt: {new_tilt:.1f}° | X-Err: {pan_error:.0f}px")

            else:
                # No face detected (or face too small)
                cv2.putText(frame_display, "No Face Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("No suitable face detected.")
                
            # 6. Display the Frame (Resized for viewing convenience)
            frame_display_resized = cv2.resize(frame_display, (960, 540))
            cv2.imshow('Live PID Tracking', frame_display_resized)

            # Exit condition: Press 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05) # Small delay to regulate loop speed
            
        except Exception as e:
            print(f"An unexpected error occurred in the loop: {e}")
            break
            
    cv2.destroyAllWindows()


# --- 5. PROGRAM ENTRY ---

if __name__ == '__main__':
    # 1. Load the required models and classifiers
    load_resources()

    # 2. Initialize the camera
    initialize_picam2()

    # 3. Initialize the servo driver
    servo_kit = initialize_pca9685()
        
    try:
        face_tracking_loop(servo_kit)
    except KeyboardInterrupt:
        print("\nTracking stopped by user (Ctrl+C).")
    finally:
        picam2.stop()  # Stop the camera feed
        cv2.destroyAllWindows()
        print("Program terminated. Resources released.")