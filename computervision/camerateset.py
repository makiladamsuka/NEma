import cv2
from picamera2 import Picamera2
import numpy as np
import time


picam2 = Picamera2()
    
# Configure the camera for a fast preview stream
config = picam2.create_preview_configuration(main={"size": (1920, 1080)}) 
picam2.configure(config)

# FIX: The AfMode control has been removed as it causes errors on fixed-focus cameras (like IMX219)
# picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) 

picam2.start()
print("Camera stream started. Press 'q' to quit.")

# Give the camera a moment to initialize
time.sleep(1) 

while True:
    # 3. Capture the frame as a NumPy array
    # This is the bridge between the camera hardware and OpenCV.
    frame = picam2.capture_array()
    
    frame = cv2.rotate(frame, cv2.ROTATE_180) 
    cv2.imshow("PiCamera2 and OpenCV Feed", frame)

    # 5. Handle key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
print("Camera stopped and windows closed.")