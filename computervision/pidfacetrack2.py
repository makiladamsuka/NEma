import cv2
from picamera2 import Picamera2
import time
from simple_pid import PID
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import sys # Added for graceful exit handling

# --- 1. CONFIGURATION PARAMETERS ---

# Camera Settings
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Servo Hardware Setup
I2C_ADDRESS = 0x40  # Default I2C address for PCA9685
PAN_CHANNEL = 1     # PCA9685 channel for horizontal servo
TILT_CHANNEL = 0    # PCA9685 channel for vertical servo
PAN_CENTER = 90     # Initial/Resting angle for Pan
TILT_CENTER = 90    # Initial/Resting angle for Tilt

# PID Controller Tuning (!!! THESE ARE STARTER VALUES - YOU MUST TUNE THEM !!!)
# Kp, Ki, Kd for Pan (Horizontal)
PAN_Kp, PAN_Ki, PAN_Kd = 3/100, .01, 0.5/10
# Kp, Ki, Kd for Tilt (Vertical)
TILT_Kp, TILT_Ki, TILT_Kd = 3/100, .01, 0.5/10

# --- NEW: SMOOTHING FACTOR (Like the video) ---
# This controls how "fast" the servo moves to its target.
# Smaller value = Slower, Smoother (e.g., 0.05)
# Larger value = Faster, Jerkier (e.g., 0.2)
SMOOTHING_FACTOR = 0.02

# --- CHANGED: PID_MAX_OFFSET ---
# This is now the maximum *offset angle* the PID can request (e.g., +/- 60 degrees from center)
PID_MAX_OFFSET = 60


# --- 2. HARDWARE & PID INITIALIZATION ---

try:
    # I2C Bus Setup
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50  # Standard frequency for hobby servos

    # Servo Setup
    pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
    tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

    # Move servos to center position
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


# PID Controller Setup
PAN_SETPOINT = FRAME_WIDTH / 2    # Horizontal center of the frame
TILT_SETPOINT = FRAME_HEIGHT / 2  # Vertical center of the frame

pan_pid = PID(PAN_Kp, PAN_Ki, PAN_Kd, setpoint=PAN_SETPOINT)
pan_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET) # <-- CHANGED

tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
tilt_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET) # <-- CHANGED


# --- 3. CAMERA AND CASCADE INITIALIZATION ---

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT) 
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1.0) # Warm-up time
print(f"Picamera2 started at {FRAME_WIDTH}x{FRAME_HEIGHT}.")


# --- 4. MAIN CONTROL LOOP ---

def cleanup_and_exit():
    """Stops the camera, resets servos, and closes OpenCV windows."""
    print("\nStopping camera and resetting servos...")
    picam2.stop()
    cv2.destroyAllWindows()
    # Reset servos to center before exiting
    pan_servo.angle = PAN_CENTER 
    tilt_servo.angle = TILT_CENTER
    sys.exit(0)

try:
    while True:
        # Read the frame from the camera
        frame = picam2.capture_array()
        
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(80, 80)
        )

        # --- LOGIC CHANGE ---
        # 1. Define *target* angles. Default to center.
        target_pan_angle = PAN_CENTER
        target_tilt_angle = TILT_CENTER

        # Only proceed with tracking if a face is found
        if len(faces) > 0:
            # Target the first (and usually largest) face
            (x, y, w, h) = faces[0]
            
            # Calculate the center of the detected face (The PID INPUT)
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # --- PID Calculation ---
            
            # 2. PID calculates a *target offset* from the center (e.g., -15 degrees)
            pan_offset = pan_pid(face_center_x)
            tilt_offset = tilt_pid(face_center_y)
            
            # 3. Calculate the final *target angle*
            # Pan: Positive offset means face is RIGHT (high X), so pan RIGHT (higher angle) -> Add
            # Tilt: Positive offset means face is DOWN (high Y), so tilt UP (higher angle) -> Add
            # (Note: Your original code had different signs, I've adjusted to what seems more intuitive
            #  but you may need to flip the + and - signs here)
            target_pan_angle = PAN_CENTER + pan_offset  
            target_tilt_angle = TILT_CENTER - tilt_offset 
            
            # --- Drawing and Visualization (for the face) ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)

        # --- NEW: Smoothing logic (runs every frame) ---
        # 4. Apply the smoothing filter (linear interpolation)
        # The new "current" angle moves *part way* to the "target" angle each frame.
        current_pan_angle = (target_pan_angle * SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - SMOOTHING_FACTOR))
        current_tilt_angle = (target_tilt_angle * SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - SMOOTHING_FACTOR))
        
        # 5. Clamp angles *after* smoothing
        current_pan_angle = max(0, min(180, current_pan_angle))
        current_tilt_angle = max(0, min(180, current_tilt_angle))

        # 6. Move the Servos
        pan_servo.angle = current_pan_angle
        tilt_servo.angle = current_tilt_angle

        # --- Drawing (runs every frame) ---
        # Draw red dot at the center of the frame (The Setpoint)
        cv2.circle(frame, (int(PAN_SETPOINT), int(TILT_SETPOINT)), 5, (0, 0, 255), -1)
        
        # Display the output frame
        cv2.imshow('PiCam Face Tracking', frame)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    cleanup_and_exit()