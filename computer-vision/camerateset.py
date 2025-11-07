import cv2
from picamera2 import Picamera2
import numpy as np
import time

# 1. Initialize Picamera2
# Analogy: This is like turning on the camera device and configuring its sensor settings.
picam2 = Picamera2()

# 2. Configure the camera stream
# Setting the resolution and the pixel format
# 'RGB888' format is often preferred for compatibility with OpenCV
camera_config = picam2.create_video_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
)

# Configure and start the camera
picam2.configure(camera_config)
picam2.start()

print("Camera stream started. Press 'q' to quit.")

# Give the camera a moment to initialize
time.sleep(1) 

while True:
    # 3. Capture the frame as a NumPy array
    # This is the bridge between the camera hardware and OpenCV.
    frame = picam2.capture_array()
    
    # Optional: OpenCV processing here (e.g., face detection, color manipulation)
    # Convert from RGB (Picamera2 default) to BGR (OpenCV's preferred color order)
    # Note: Using 'RGB888' in the config usually avoids BGR/RGB issues, but sometimes it is necessary.
    # processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 4. Display the processed frame
    cv2.imshow("PiCamera2 and OpenCV Feed", frame)

    # 5. Handle key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()
print("Camera stopped and windows closed.")