import cv2
from picamera2 import Picamera2 # The new library for Raspberry Pi camera
import time

# 1. Load the pre-trained classifier
# Using cv2.data.haarcascades ensures the path is correct if OpenCV was installed correctly.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Initialize the Picamera2 object and configure it
picam2 = Picamera2()
# Set a smaller size for faster processing, which is crucial on a Pi
picam2.preview_configuration.main.size = (1280 , 720) 
picam2.preview_configuration.main.format = "RGB888" # Define the output format
picam2.configure("preview")
picam2.start()

# Give the camera a moment to warm up
time.sleep(1.0)

print("Starting Face Detection Stream...")

# Main loop to continuously process frames
while True:
    # 3. Read the frame from the camera
    # capture_array() reads the frame as a NumPy array (which OpenCV needs)
    frame = picam2.capture_array()

    # 4. Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1, # How much the image size is reduced at each image scale
        minNeighbors=5,  # How many neighbors each candidate rectangle should have
        minSize=(40, 40) # Minimum face size to be considered a face
    )

    # 6. Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Changed color to Green for visibility

    # 7. Display the output
    cv2.imshow('PiCam Face Detection', frame)

    # 8. Check for 'q' key press to quit
    # waitKey(1) means wait for 1 millisecond for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Clean up resources
picam2.stop()
cv2.destroyAllWindows()