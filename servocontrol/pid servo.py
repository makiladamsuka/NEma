import time
from simple_pid import PID
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import sys 

# --- 1. CONFIGURATION PARAMETERS ---
# Frame dimensions are kept as they define the coordinate system for PID.
FRAME_WIDTH, FRAME_HEIGHT = 640, 480 

# Servo Hardware Setup
I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90

# PID Tuning
PAN_Kp, PAN_Ki, PAN_Kd = 3/100, .001, .5/100
TILT_Kp, TILT_Ki, TILT_Kd = 3/100, .001, .5/100

SMOOTHING_FACTOR = .008
PID_MAX_OFFSET = 60
MAX_SEARCH_FRAMES = 50

# --- 2. GLOBAL STATE VARIABLES (MUST BE TRACKED ACROSS FRAMES) ---
current_pan_angle = PAN_CENTER
current_tilt_angle = TILT_CENTER
last_face_x = FRAME_WIDTH / 2
last_face_y = FRAME_HEIGHT / 2
IS_SEARCHING = False
search_frame_counter = 0

# --- 3. PID INITIALIZATION (Global) ---
PAN_SETPOINT = FRAME_WIDTH / 2
TILT_SETPOINT = FRAME_HEIGHT / 2

pan_pid = PID(PAN_Kp, PAN_Ki, PAN_Kd, setpoint=PAN_SETPOINT)
pan_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)

tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
tilt_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)


# ====================================================================
## 🔄 THE REFACTORED SERVO CONTROL FUNCTION
# Accepts (center_x, center_y) or None
# ====================================================================

def track_face_and_calculate_angles(face_center_coords):
    """
    Calculates the required servo angles based on the detected face center 
    or initiates a search pattern if no face is found.

    Args:
        face_center_coords (tuple or None): 
            If face found: (face_center_x, face_center_y).
            If no face: None.

    Returns:
        tuple: (pan_angle_in_degrees, tilt_angle_in_degrees, tracking_state_text)
    """
    # ⚠️ These variables must be global to maintain their state between calls
    global current_pan_angle, current_tilt_angle
    global last_face_x, last_face_y
    global IS_SEARCHING, search_frame_counter
    global pan_pid, tilt_pid
    
    # Defaults
    pan_offset, tilt_offset = 0, 0
    tracking_state_text = "Idle"

    # --- TRACKING LOGIC ---
    if face_center_coords is not None:
        IS_SEARCHING = False 
        search_frame_counter = 0
        
        # Determine face center from the input coordinates
        (face_center_x, face_center_y) = face_center_coords 
        
        # Update Last Known Position
        last_face_x = face_center_x
        last_face_y = face_center_y

        # PID Input is the current face center
        pan_offset = pan_pid(face_center_x)
        tilt_offset = tilt_pid(face_center_y)
        
        tracking_state_text = "Tracking"
        
    else: # NO FACE DETECTED
        IS_SEARCHING = True
        
        if search_frame_counter < MAX_SEARCH_FRAMES:
            # Use the LAST KNOWN position for PID input (Momentum Search)
            pan_offset = pan_pid(last_face_x)
            tilt_offset = tilt_pid(last_face_y)
            
            tracking_state_text = f"Searching ({search_frame_counter}/{MAX_SEARCH_FRAMES})"
            search_frame_counter += 1
            
        else:
            # Give up and go to center (pan_offset, tilt_offset remain 0)
            tracking_state_text = "Idle"

    # --- Servo Angle Calculation ---
    # Convert PID offset (pixel error) into angle adjustment
    target_pan_angle = PAN_CENTER + pan_offset  
    target_tilt_angle = TILT_CENTER - tilt_offset  

    # Apply Smoothing (Linear Interpolation)
    current_pan_angle = (target_pan_angle * SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - SMOOTHING_FACTOR))
    current_tilt_angle = (target_tilt_angle * SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - SMOOTHING_FACTOR))
    
    # Clamp angles
    current_pan_angle = max(0, min(180, current_pan_angle))
    current_tilt_angle = max(0, min(180, current_tilt_angle))

    # Return the new angles and tracking state information
    return current_pan_angle, current_tilt_angle, tracking_state_text


# ====================================================================
## ⚙️ HARDWARE & MAIN LOOP
# Simplified Main Loop for Demonstration/Integration
# ====================================================================

# --- HARDWARE INITIALIZATION ---
try:
    # Initialize I2C and PCA9685 board
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50 

    # Initialize Servos
    pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
    tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

    # Move servos to center
    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    print(f"PCA9685 initialized. Servos set to {PAN_CENTER}°, {TILT_CENTER}° tilt.")

except ValueError:
    print("Error: Could not find PCA9685 at the specified I2C address.")
    sys.exit(1)
except ImportError as e:
    print(f"Error: Required library not found ({e}). Ensure adafruit-blinka and adafruit-circuitpython-pca9685 are installed.")
    sys.exit(1)


# --- CLEANUP FUNCTION ---
def cleanup_and_exit():
    """Resets servos before exiting."""
    print("\nResetting servos...")
    pan_servo.angle = PAN_CENTER 
    tilt_servo.angle = TILT_CENTER
    time.sleep(0.5) 
    sys.exit(0)

# --- SIMULATED MAIN CONTROL LOOP ---

    """
    Simulates an external script providing face coordinates to the tracking function.
    
    In a real application, you would replace the loop structure below with the 
    actual data stream (e.g., from a network socket or shared memory) from your 
    external computer vision program.
    """
    print("\nStarting Tracking Demo (No Camera Input - Simulating External Coordinates)")
    
    # === SIMULATION SETUP ===
    # Define a sequence of target positions to test tracking, losing, and re-acquiring.
    
    # 1. Start far left (x=50 is far left in a 640x480 frame)
    coords_list = [(50, 200)] * 50  
    # 2. Move to the center
    coords_list += [(320, 240)] * 100
    # 3. Move far right and up
    coords_list += [(600, 100)] * 50
    # 4. Lose target (None should trigger the search/momentum logic)
    coords_list += [None] * 100 
    # 5. Find target again (center)
    coords_list += [(320, 240)] * 50

    try:
        frame_count = 0
        for coords in coords_list:
            frame_count += 1
            
            # 1. --- CALL THE CORE TRACKING FUNCTION ---
            # Pass only the derived coordinates (center_x, center_y) or None
            new_pan_angle, new_tilt_angle, tracking_state = track_face_and_calculate_angles(coords)

            # 2. --- ACTUATE SERVOS ---
            pan_servo.angle = new_pan_angle
            tilt_servo.angle = new_tilt_angle

            # 3. --- LOGGING ---
            face_status = f"Tracking target at: {coords}" if coords else "Target lost. Initiating search."
            
            print(f"Frame {frame_count}: {tracking_state} | Input: {face_status}")
            print(f"  -> Servo Angles: PAN={new_pan_angle:.2f}°, TILT={new_tilt_angle:.2f}°")
            
            time.sleep(0.05) # Simulate frame processing time (20 FPS)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    finally:
        cleanup_and_exit()


if __name__ == '__main__':
    run_tracking_demo()