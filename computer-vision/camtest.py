import cv2
import numpy as np
from picamera2 import Picamera2 # ?? Import the official library

# 1. Initialize the Picamera2 object
picam2 = Picamera2()

# 2. Configure the camera for video streaming
# 'XRGB8888' is a good format for seamless conversion to OpenCV arrays
config = picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(config)

# 3. Start the camera stream
picam2.start()

# Loop to continuously read and display frames
while True:
    # 4. Capture the frame as a NumPy array
    # This is the crucial step: picamera2 gets the image, and it's ready for OpenCV
    frame = picam2.capture_array()

    # Optional: Display a circle on the frame using an OpenCV function
    # This proves the frame is correctly loaded into OpenCV
    height, width, _ = frame.shape
    cv2.circle(frame, (width//2, height//2), 50, (0, 255, 0), 2) # Green circle in the middle

    # 5. Display the resulting frame using OpenCV
    cv2.imshow('PiCamera2 + OpenCV Feed', frame)

    # 6. Wait 1ms for key press, if 'q' is pressed, exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
picam2.stop()
cv2.destroyAllWindows()