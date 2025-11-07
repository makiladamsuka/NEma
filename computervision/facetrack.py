import cv2

# Analogy: This is like loading a blueprint or a detailed instruction manual
# that tells the computer exactly what features make up a 'frontal face'.
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the default webcam (0 is usually the built-in webcam)
video_capture = cv2.VideoCapture(0)

# Check if the webcam was opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Webcam feed opened. Press 'q' to quit.")

while True:
    # Read a single frame from the video stream
    result, video_frame = video_capture.read()
    
    # If the frame wasn't read correctly, break the loop
    if not result:
        break

    # Convert the frame to grayscale for faster processing
    # The computer is looking for changes in lightness/darkness, so color is unnecessary.
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1, # How much the image size is reduced at each image scale
        minNeighbors=5,  # How many neighbors each candidate rectangle should have
        minSize=(40, 40) # Minimum face size to be considered a face
    )

    # Draw a rectangle (bounding box) around each detected face
    # The 'faces' variable returns a list of coordinates (x, y, width, height)
    for (x, y, w, h) in faces:
        # Draw the rectangle: (image, top-left point, bottom-right point, color, thickness)
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Display the processed frame in a window
    cv2.imshow("Live Face Detection", video_frame)

    # Stop the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up by releasing the webcam and closing all windows
video_capture.release()
cv2.destroyAllWindows()