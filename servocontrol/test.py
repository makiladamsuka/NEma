import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import sys

# --- Import your puppeteer file ---
import head_movements

# --- Servo Hardware Setup (Same as your main script) ---
I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90

# --- Setup I2C and Servos ---
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50 

    pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
    tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    
    print("PCA9685 initialized.")

except Exception as e:
    print(f"Error setting up hardware: {e}")
    sys.exit(1)

# --- Initialize the Puppeteer ---
head_movements.init(pan_servo, tilt_servo)

# --- Main Test Loop ---
try:
    print("Starting head movement test in 3 seconds...")
    time.sleep(3)

    # --- CHANGE IS HERE ---
    # 1. Set the name of the one move you want to test
    move_to_test = 'happy'  # <-- REPLACE 'happy' with your move name!

    print(f"--- Testing move: '{move_to_test}' ---")
    head_movements.start_move(move_to_test)
    
    # 2. We keep this loop to make the move happen
    # This update loop is still needed
    while head_movements.is_active():
        head_movements.update()
        time.sleep(0.02) # Don't spin the CPU too fast
        
    print(f"Finished '{move_to_test}'.")
    # --- END OF CHANGE ---

    print("--- Test complete! ---")

except KeyboardInterrupt:
    print("\nTest stopped by user.")

