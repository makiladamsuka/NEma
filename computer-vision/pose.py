import cv2
import mediapipe as mp 

WIDTH = 640
HEIGHT = 480

# --- Setup ---
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Initialize the Pose model
# (The arguments set the complexity and confidence thresholds)
pose = mp.solutions.pose.Pose(False, True, True, 0.5, 0.5, 0.5, 0.5)

# Initialize the drawing utility
mpDraw = mp.solutions.draw_utils
# Get the connections needed to draw the 'skeleton'
mpPose = mp.solutions.pose 


while True:
    _, frame = cam.read()
    
    # 1. Prepare the frame for MediaPipe
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Process the frame (Find the pose)
    results = pose.process(frame_rgb)
    
    # 3. DRAW THE POSE ON THE FRAME (NEW STEP)
    if results.pose_landmarks:
        # This draws the dots (landmarks) and lines (connections) 
        # onto the original BGR 'frame'
        mpDraw.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mpPose.POSE_CONNECTIONS,
            # You can optionally customize colors and thickness here
        )
    
    # 4. Display the frame
    cv2.imshow("Webcam Feed - Pose Tracker", frame)
    
    # 5. Exit condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
# --- Cleanup ---
cam.release()
cv2.destroyAllWindows()