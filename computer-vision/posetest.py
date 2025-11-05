import cv2 
import mediapipe as mp

# --- CONFIGURATION ---
# 1. CHANGE THIS FILE PATH to your image file
IMAGE_FILE_PATH = "your_image_file.jpg" 
# Example: "C:/Users/User/Pictures/my_pose_pic.png"

# --- INITIALIZATION ---
# Initialize the Pose model
pose = mp.solutions.pose.Pose(False, True, True, 0.5, 0.5, 0.5, 0.5)
mpDraw = mp.solutions.draw_utils
mpPose = mp.solutions.pose

# --- IMAGE PROCESSING ---

# 1. Read the image from the file path
frame = cv2.imread(IMAGE_FILE_PATH)

# Check if the image was loaded successfully
if frame is None:
    print(f"ERROR: Could not load image from path: {IMAGE_FILE_PATH}")
else:
    # 2. Convert the image color space (BGR to RGB) for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Process the image (Find the pose)
    results = pose.process(frame_rgb)
    
    # 4. Draw the pose landmarks on the original BGR frame
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mpPose.POSE_CONNECTIONS,
            # Customize drawing appearance (optional)
            mpDraw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), # Landmarks in Blue
            mpDraw.DrawingSpec(color=(0,255,0), thickness=2) # Connections in Green
        )
    
    # 5. Display the result
    cv2.imshow("Pose Estimation Result", frame)
    
    # Wait indefinitely until a key is pressed (unlike the webcam's cv2.waitKey(1))
    cv2.waitKey(0) 

# --- CLEANUP ---
cv2.destroyAllWindows()