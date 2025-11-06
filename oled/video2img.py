import cv2
import os
from PIL import Image

# --- CONFIGURATION ---
VIDEO_FILE = "video3.mp4"   # <--- 💡 CHANGE THIS to your video file name
OUTPUT_FOLDER = "oled_frames3"         # Folder where the small images will be saved
OLED_WIDTH = 128
OLED_HEIGHT = 128
FRAME_SKIP = 3                       # Only process every 3rd frame (adjust for speed/choppiness)
# ---------------------

def preprocess_video():
    # 1. Setup
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_FILE}")
        return

    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Starting processing of {total_frames} frames...")

    # 2. Frame Processing Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        
        # Skip frames to reduce the final file size and increase playback speed
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # --- Processing Steps ---
        
        # a. Resize to OLED resolution
        resized_frame = cv2.resize(frame, (OLED_WIDTH, OLED_HEIGHT), interpolation=cv2.INTER_AREA)

        # b. Convert to Grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # c. Convert to pure Black/White (Monochrome)
        # This uses a threshold (127) to make everything either pure black (0) or pure white (255)
        _, monochrome_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        # d. Convert to PIL Image object for easy saving in monochrome mode
        pil_image = Image.fromarray(monochrome_frame)
        
        # 3. Save the Frame
        # Format the filename with leading zeros (e.g., frame_0001.png)
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.png")
        # Save in '1' mode (1-bit black/white) for maximum efficiency
        pil_image.save(filename, format="PNG", optimize=True) 
        
        saved_count += 1
        
        if saved_count % 50 == 0:
            print(f"Processed and saved {saved_count} frames...")

    print("-" * 30)
    print(f"Finished! Total frames saved: {saved_count}")
    print(f"Now transfer the '{OUTPUT_FOLDER}' folder to your Raspberry Pi.")
    
    cap.release()

if __name__ == "__main__":
    preprocess_video()