import cv2
import numpy as np
import time
from picamera2 import Picamera2 

WIDTH, HEIGHT = 640, 480
CENTERX, CENTERY = WIDTH // 2, HEIGHT // 2

face_classifier = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={"format": 'RGB888', "size": (WIDTH, HEIGHT)} 
)
picam2.configure(camera_config)
picam2.start()

# Give the camera a moment to initialize
time.sleep(1)

print("PiCamera feed opened. Press 'q' to quit.")

while True:
    video_frame = picam2.capture_array()
    
    
    if video_frame is None or video_frame.size == 0:
        print("Error: Could not capture frame from Picamera2.")
        break
    
    
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(40, 40)
    )
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face
        
            
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        offset_x = face_center_x - CENTERX
        offset_y = face_center_y - CENTERY
        
        print(f"Face Offset - X: {offset_x}, Y: {offset_y}")
    
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


picam2.stop() 
print("Cleanup complete. Camera stopped.")