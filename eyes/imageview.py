# Import necessary parts
import cv2            # OpenCV for display and rotation
from PIL import Image # Still needed for image loading and cropping
import time
import os
import glob 
import numpy as np    # OpenCV works best with NumPy arrays

# --- CONFIGURATION ---
FRAMES_FOLDER = "oled_frames" 
# ---------------------

def display_preprocessed_video():
    # 1. Setup
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_FOLDER, "*.png")))
    
    if not frame_paths:
        print(f"Error: No image files found in the folder: {FRAMES_FOLDER}.")
        return

    # 2. Load all frames into memory (NumPy arrays)
    loaded_np_arrays = []
    print("Loading and converting all images (128x128) into memory...")
    for frame_file in frame_paths:
        # Load, convert to 1-bit monochrome, and crop the top 128x64 portion
        full_img = Image.open(frame_file).convert('1')
        top_128x64_portion = full_img.crop((0, 0, 128, 64))
        
        # Convert to an 8-bit NumPy array immediately for easier processing later
        img_np = np.array(top_128x64_portion).astype(np.uint8) * 255
        loaded_np_arrays.append(img_np)
        
    print("Image loading complete. Starting display loop. Press 'q' in a window to quit.")
    
    # 3. Display Loop
    while True:
        start_time_total = time.monotonic()
        
        for partial_img_np in loaded_np_arrays:
            
            # --- Image Splitting Logic (Vertical Split) ---
            # Now splitting the NumPy array (128x64)
            # The coordinates are [row_start:row_end, col_start:col_end]
            
            # Left Half (64x64): All rows (0 to 64), first half of columns (0 to 64)
            left_half = partial_img_np[:, 0:64] 
            
            # Right Half (64x64): All rows (0 to 64), second half of columns (64 to 128)
            right_half = partial_img_np[:, 64:128]
            
            # --- Rotation Step using cv2.rotate() ---
            # Rotate both image pieces 90 degrees clockwise
            rotated_left = cv2.rotate(left_half, cv2.ROTATE_90_CLOCKWISE)
            rotated_right = cv2.rotate(right_half, cv2.ROTATE_90_CLOCKWISE)

            # --- Display using OpenCV ---
            cv2.imshow("Left OLED View (Rotated)", rotated_left)
            cv2.imshow("Right OLED View (Rotated)", rotated_right)

            # --- Loop Control ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return 

        # Calculate and print actual FPS for one loop
        end_time_total = time.monotonic()
        total_duration = end_time_total - start_time_total
        
        if total_duration > 0:
            actual_fps = len(loaded_np_arrays) / total_duration
            print(f"One loop complete: {len(loaded_np_arrays)} frames displayed in {total_duration:.2f}s. Actual FPS: {actual_fps:.2f}")

if __name__ == "__main__":
    try:
        display_preprocessed_video()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Display windows closed.")