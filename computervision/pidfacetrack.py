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

# PID Controller Tuning (THESE MUST BE TUNED FOR YOUR SYSTEM!)
# Kp, Ki, Kd for Pan (Horizontal)
PAN_Kp, PAN_Ki, PAN_Kd = 2/1000, .01, 0.5/1000
# Kp, Ki, Kd for Tilt (Vertical)
TILT_Kp, TILT_Ki, TILT_Kd = 2/1000, .01, 0.5/1000
# Max degree change per frame to limit servo speed
MAX_ANGLE_CHANGE = 0.4


# --- 2. HARDWARE & PID INITIALIZATION ---

try:
    # I2C Bus Setup
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50  # Standard frequency for hobby servos

    # Servo Setup
    # min_pulse/max_pulse may need adjustment for specific servos
    pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
    tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

    # Move servos to center position
    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    current_pan_angle = PAN_CENTER
    current_tilt_angle = TILT_CENTER

    print(f"PCA9685 initialized. Servos set to {PAN_CENTER}° pan, {TILT_CENTER}° tilt.")

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
pan_pid.output_limits = (-MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)

tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
tilt_pid.output_limits = (-MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)


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

        # Detect faces (Optimized for Raspberry Pi)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(80, 80)
        )

        # Only proceed with tracking if a face is found
        if len(faces) > 0:
            # Target the first (and usually largest) face
            (x, y, w, h) = faces[0]
            
            # Calculate the center of the detected face (The PID INPUT)
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # --- PID Calculation and Servo Movement ---
            
            # Calculate correction based on error between face center and frame center
            pan_correction = pan_pid(face_center_x)
            tilt_correction = tilt_pid(face_center_y)

            # Update angles: 
            # Pan: Positive correction means face is RIGHT (high X), so pan LEFT (lower angle) -> Subtract
            # Tilt: Positive correction means face is DOWN (high Y), so tilt UP (higher angle) -> Add
            current_pan_angle += pan_correction 
            current_tilt_angle -= tilt_correction 
            
            # Clamp angles to physical limits (0-180 degrees)
            current_pan_angle = max(0, min(180, current_pan_angle))
            current_tilt_angle = max(0, min(180, current_tilt_angle))

            # Move the Servos
            pan_servo.angle = current_pan_angle
            tilt_servo.angle = current_tilt_angle

            # --- Drawing and Visualization ---

            # Draw green rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw blue dot at the center of the face (The Target)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)

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