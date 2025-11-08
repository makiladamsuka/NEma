import time
from computervision.emotiondetection.emocapture import run_emotion_detector 
from adafruit_servokit import ServoKit 

# Import the PID controller from the simple-pid library
from simple_pid import PID 

# --- Tracking Configuration ---
FRAME_CENTER_X = 1920 // 2
FRAME_CENTER_Y = 1080 // 2

# **PID GAIN TUNING**: These values must be adjusted for your setup!
# Kp (Proportional) is the main driver. Start with Ki=0, Kd=0.
# Tune Kp until it oscillates, then set Kp to about half of that value.
PID_PAN_GAINS = (0.012, 0.0001, 0.0005)   # (Kp, Ki, Kd) - Example starting values
PID_TILT_GAINS = (0.010, 0.0001, 0.0004) # Tune these separately

# Servo Limits (in degrees)
PAN_MIN = 30  
PAN_MAX = 150 
TILT_MIN = 60 
TILT_MAX = 120 

# Global servo angles
current_pan_angle = 90
current_tilt_angle = 90

# --- PCA9685 Initialization (unchanged) ---
def initialize_pca9685():
    """Initializes the PCA9685 module and ServoKit."""
    global current_pan_angle, current_tilt_angle
    print("Initializing PCA9685...")
    try:
        kit = ServoKit(channels=16) 
        kit.servo[0].angle = current_pan_angle
        kit.servo[1].angle = current_tilt_angle
        print("PCA9685 initialized. Servos centered.")
        return kit
    except Exception as e:
        print(f"Error initializing PCA9685/I2C: {e}")
        print("Please ensure I2C is enabled and the module is wired correctly.")
        exit()

# --- Error Calculation ---
def calculate_errors(face_coords):
    """Calculates the Pan (X) and Tilt (Y) errors (pixels from center)."""
    x, y, w, h = face_coords
    
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Pan Error (X): Face on the right -> Positive error
    pan_error = face_center_x - FRAME_CENTER_X
    # Tilt Error (Y): Face on the bottom -> Positive error
    tilt_error = face_center_y - FRAME_CENTER_Y
    
    return pan_error, tilt_error

# --- Servo Update ---
def update_servos(kit, pan_adjustment, tilt_adjustment):
    """
    Applies the calculated adjustment (change in angle) to the current servo angles.
    """
    global current_pan_angle, current_tilt_angle
    
    # PID output is the required change to move TOWARDS the target.
    new_pan_angle = current_pan_angle + pan_adjustment 
    new_tilt_angle = current_tilt_angle + tilt_adjustment
    
    # Constrain angles within their min/max limits
    new_pan_angle = max(PAN_MIN, min(PAN_MAX, new_pan_angle))
    new_tilt_angle = max(TILT_MIN, min(TILT_MAX, new_tilt_angle))

    # Update the servos if the angle has changed significantly
    if abs(new_pan_angle - current_pan_angle) > 0.1:
        kit.servo[0].angle = new_pan_angle
        current_pan_angle = new_pan_angle
        
    if abs(new_tilt_angle - current_tilt_angle) > 0.1:
        kit.servo[1].angle = new_tilt_angle
        current_tilt_angle = new_tilt_angle
    
    return new_pan_angle, new_tilt_angle

# --- Main Tracking Loop ---
def face_tracking_loop(kit):
    print("Setting up PID controllers...")
    
    # 1. Initialize the Pan PID controller
    pid_pan = PID(
        Kp=PID_PAN_GAINS[0], Ki=PID_PAN_GAINS[1], Kd=PID_PAN_GAINS[2], 
        setpoint=0, 
        sample_time=0.05, # Matches the loop sleep time
        output_limits=(-5.0, 5.0) # Limits max angle change per loop (important for safety)
    )
    
    # 2. Initialize the Tilt PID controller
    pid_tilt = PID(
        Kp=PID_TILT_GAINS[0], Ki=PID_TILT_GAINS[1], Kd=PID_TILT_GAINS[2], 
        setpoint=0, 
        sample_time=0.05,
        output_limits=(-5.0, 5.0)
    )
    
    print("\nStarting PID-controlled face tracking...")
    
    while True:
        emotion, face_coords = run_emotion_detector() 
        
        if face_coords is not None:
            # 1. Calculate the error (distance from center in pixels)
            pan_error, tilt_error = calculate_errors(face_coords)
            
            # 2. Get adjustment from PID controllers
            
            # Pan: A positive error (face on right) must produce a NEGATIVE angle change.
            # We invert the error when passing it to the PID to get the correct output.
            pan_adjustment = pid_pan(-pan_error) 
            
            # Tilt: A positive error (face on bottom) must produce a POSITIVE angle change.
            # We pass the error directly.
            tilt_adjustment = pid_tilt(tilt_error) 
            
            # 3. Apply the PID output to the servos
            new_pan, new_tilt = update_servos(kit, pan_adjustment, tilt_adjustment)
            
            print(f"Tracking: {emotion.upper()} | Pan: {new_pan:.1f}° | Tilt: {new_tilt:.1f}° | X-Err: {pan_error:.0f}px")

        else:
            print("No suitable face detected.")

        # Sleep time matches the PID's sample_time
        time.sleep(0.05) 

# --- Program Entry ---
if __name__ == '__main__':
    servo_kit = initialize_pca9685()
    
    try:
        face_tracking_loop(servo_kit)
    except KeyboardInterrupt:
        print("\nTracking stopped by user (Ctrl+C).")
    finally:
        print("Servo tracking program terminated.")
        