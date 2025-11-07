import cv2 
import mediapipe as mp
import numpy as np 

WIDTH = 640
HEIGHT = 480

# --- SETUP: Camera and Pose Model ---

# Initialize the webcam, checking for potential errors
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("ERROR: Could not open the webcam (index 0). Trying index 1...")
    cam = cv2.VideoCapture(1) # Try next index
    if not cam.isOpened():
        print("FATAL ERROR: Failed to open webcam on indices 0 and 1. Please check camera connection and permissions.")
        # Analogy: The CCTV operator can't turn on the camera!
        exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT) 

# Initialize the Pose model
pose = mp.solutions.pose.Pose(False, True, True, 0.5, 0.5, 0.5, 0.5)
mpDraw = mp.solutions.draw_utils
mpPose = mp.solutions.pose

# --- HELPER FUNCTION: Angle Calculation ---
def calculate_angle(a, b, c):
    """Calculates the 2D angle (in degrees) between three points (a, b, c) 
    with b being the vertex (e.g., Elbow).
    Uses the screen (x, y) coordinates for calculation.
    """
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow (Vertex)
    c = np.array(c) # Wrist
    
    # Calculate angles using 2D arctan2
    angle_ab = np.arctan2(a[1] - b[1], a[0] - b[0]) 
    angle_cb = np.arctan2(c[1] - b[1], c[0] - b[0])
    
    # Calculate difference, convert to degrees, and ensure 0-180 range
    angle_rad = np.abs(angle_ab - angle_cb)
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg > 180.0:
        angle_deg = 360 - angle_deg
        
    return angle_deg

# --- NEW VARIABLES FOR TRACKING MOTION ---
# Store the previous X-coordinate of the wrist to detect side-to-side motion
prev_wrist_x = 0 
# Counter to track how many frames the "wave motion" has been happening
wave_counter = 0
# Threshold for X-coordinate change to count as motion (normalized value 0 to 1)
MOTION_THRESHOLD = 0.02 
# Number of frames required to confirm a wave (adjust based on your framerate)
WAVE_FRAMES_REQUIRED = 15 
# --- END NEW VARIABLES ---
    
# --- MAIN LOOP ---
while True:
    success, frame = cam.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue # Skip if frame reading failed
        
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    action_text = "Action: Searching..." # Default status

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- 1. Extract Points (Using Right Arm) ---
            # Get normalized (x, y) coordinates for the relevant joints
            shoulder = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                        landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x, 
                     landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x, 
                     landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate Elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # --- 2. WAVE DETECTION LOGIC ---
            current_wrist_x = wrist[0]
            
            # Condition 1: Arm is relatively straight (low angle) and raised (wrist Y is lower than shoulder Y)
            # A straight arm is usually > 140 degrees. Y-coordinate 0 is the top of the screen.
            is_arm_raised_and_straight = (angle > 140) and (wrist[1] < shoulder[1]) 

            # Condition 2: Check for significant horizontal movement (side-to-side)
            is_moving_horizontally = abs(current_wrist_x - prev_wrist_x) > MOTION_THRESHOLD

            if is_arm_raised_and_straight and is_moving_horizontally:
                # If conditions met, increase the counter
                wave_counter += 1
                action_text = f"Action: Waving Motion Detected... ({wave_counter}/{WAVE_FRAMES_REQUIRED})"
            else:
                # If motion stops or arm drops, decrease the counter slowly
                wave_counter = max(0, wave_counter - 2) 

            # Condition 3: Check if the wave has been sustained
            if wave_counter >= WAVE_FRAMES_REQUIRED:
                action_text = "**WAVING!** 👋"
                
            # Update the previous wrist position for the next frame
            prev_wrist_x = current_wrist_x

        except Exception as e:
            # Catch errors if one of the critical landmarks is not found
            # print(f"Error during pose processing: {e}") 
            pass
            
        # Draw landmarks after all calculations
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Display the action status on the screen
    cv2.putText(frame, action_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Webcam Feed - Pose Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- CLEANUP ---
cam.release()
cv2.destroyAllWindows()