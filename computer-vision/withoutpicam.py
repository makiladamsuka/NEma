import cv2
import time

# --- Configuration ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30
# ---------------------

print(f"Attempting to open camera on index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2) 

if not cap.isOpened():
    print("ERROR! Unable to open camera.")
else:
    print("Camera successfully opened. Now attempting to set properties...")
    
    # 1. Set properties explicitly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # Verify properties (optional, but good for debugging)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested: {FRAME_WIDTH}x{FRAME_HEIGHT}@{FRAME_RATE}FPS")
    print(f"Actual: {actual_width}x{actual_height}@{int(cap.get(cv2.CAP_PROP_FPS))}FPS")
    
    # 2. Add a short delay to allow the V4L2 pipeline to settle
    print("Pausing for 1 second...")
    time.sleep(1) 
    
    # 3. Read and discard a few initial frames (clearing buffer)
    for i in range(5):
        cap.read()
        print(f"Discarding frame {i+1}...")

    # Start the main loop
    print("Starting video capture loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("CRITICAL ERROR: Can't receive frame (stream end?). Exiting ...")
            break
        
        cv2.imshow('OpenCV Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
