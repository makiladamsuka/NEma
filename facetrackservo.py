import time
# This imports the function that returns (emotion, (x, y, w, h))
from computervision.emotiondetection.emocapture import run_emotion_detector 
from adafruit_servokit import ServoKit 

# --- Servo Tracking Configuration ---
# Camera resolution used in the main program (1920, 1080)
FRAME_CENTER_X = 1920 // 2
FRAME_CENTER_Y = 1080 // 2

# Sensitivity: Controls speed of tracking (P-Control gain). Tune this!
# Smaller value = slower, smoother; Larger value = faster, risk of oscillation.
PAN_SENSITIVITY = 0.01 
TILT_SENSITIVITY = 0.01

# Servo Limits (in degrees)
PAN_MIN = 30  
PAN_MAX = 150 
TILT_MIN = 60 
TILT_MAX = 120 

# Global variables for current servo position
current_pan_angle = 90
current_tilt_angle = 90

# --- PCA9685 Initialization ---
def initialize_pca9685():
    """Initializes the PCA9685 module and ServoKit."""
    print("Initializing PCA9685...")
    try:
        # Assumes the default I2C address (0x40) and 16 channels
        kit = ServoKit(channels=16) 
        
        # Set initial positions for Pan (Channel 0) and Tilt (Channel 1)
        kit.servo[0].angle = current_pan_angle
        kit.servo[1].angle = current_tilt_angle
        
        print("PCA9685 initialized. Servos centered.")
        return kit
    except Exception as e:
        print(f"Error initializing PCA9685/I2C: {e}")
        print("Please ensure I2C is enabled and the module is wired correctly.")
        # Exit the program if the critical hardware component fails to load
        exit()

# --- Tracking Logic ---
def calculate_servo_adjustment(face_coordinates):
    """
    Calculates the proportional change needed for the Pan and Tilt servos 
    based on the face position relative to the frame center.
    
    The previous error of calling run_emotion_detector() here has been fixed.
    """
    # Unpack the valid coordinates passed in from the main loop
    x, y, w, h = face_coordinates
    
    # Calculate the center point of the detected face
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Calculate the error (distance from the center of the camera frame)
    error_x = face_center_x - FRAME_CENTER_X
    error_y = face_center_y - FRAME_CENTER_Y
    
    # Calculate adjustment (Proportional Control)
    # Pan: Negative error_x (face left) means positive pan_change (move right/increase angle)
    pan_change = -error_x * PAN_SENSITIVITY 
    
    # Tilt: Positive error_y (face low) means positive tilt_change (move down/increase angle)
    tilt_change = error_y * TILT_SENSITIVITY 
    
    return pan_change, tilt_change

def update_servos(kit, pan_change, tilt_change):
    """
    Applies the calculated change to the current servo angles, respecting limits.
    """
    global current_pan_angle, current_tilt_angle
    
    # Calculate new angles
    new_pan_angle = current_pan_angle + pan_change
    new_tilt_angle = current_tilt_angle + tilt_change
    
    # Constrain angles within their min/max limits
    new_pan_angle = max(PAN_MIN, min(PAN_MAX, new_pan_angle))
    new_tilt_angle = max(TILT_MIN, min(TILT_MAX, new_tilt_angle))

    # Update the servos if the angle has changed significantly (0.1 degree threshold)
    if abs(new_pan_angle - current_pan_angle) > 0.1:
        kit.servo[0].angle = new_pan_angle
        current_pan_angle = new_pan_angle
        
    if abs(new_tilt_angle - current_tilt_angle) > 0.1:
        kit.servo[1].angle = new_tilt_angle
        current_tilt_angle = new_tilt_angle
    
    return new_pan_angle, new_tilt_angle


# --- Main Tracking Loop ---
def face_tracking_loop(kit):
    """The main loop for continuously detecting and tracking the face."""
    print("Starting face tracking loop...")
    
    while True:
        # Step 1: Call the emotion detector ONCE and get results
        emotion, face_coords = run_emotion_detector() 
        
        # Step 2: Check if a face was successfully detected
        if face_coords is not None:
            # Step 3: Calculate how much to move the servos, passing the valid coordinates
            pan_change, tilt_change = calculate_servo_adjustment(face_coords)
            
            # Step 4: Send the movement command to the servos
            new_pan, new_tilt = update_servos(kit, pan_change, tilt_change)
            
            print(f"Tracking: {emotion.upper()} | Pan: {new_pan:.1f}° | Tilt: {new_tilt:.1f}°")

        else:
            print("No suitable face detected.")

        # Control the update rate (20 times per second)
        time.sleep(0.05) 

# --- Program Entry ---
if __name__ == '__main__':
    # Initialize the servo driver
    servo_kit = initialize_pca9685()
    
    try:
        # Start the tracking loop
        face_tracking_loop(servo_kit)
    except KeyboardInterrupt:
        print("\nTracking stopped by user (Ctrl+C).")
    finally:
        # Optional: Add code here to move servos to a safe/center position on exit
        print("Servo tracking program terminated.")