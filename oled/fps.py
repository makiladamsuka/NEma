# Import necessary parts
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306 
from PIL import Image
import time
import os
import glob # For easily finding all the image files
import math # Added for math.ceil/floor if needed, but not strictly required here

# --- CONFIGURATION ---
I2C_PORT = 1
I2C_ADDRESS = 0x3c 
FRAMES_FOLDER = "oled_frames" # <--- MUST MATCH the folder name you transferred
TARGET_FPS = 24 #<--- NEW: Your desired frame rate
# ---------------------

# Calculate the required time for each frame (in seconds)
# Target time per frame = 1 second / TARGET_FPS
TARGET_FRAME_TIME = 1.0 / TARGET_FPS

# Setup the connection
serial = i2c(port=I2C_PORT, address=I2C_ADDRESS)
try:
    device = ssd1306(serial)
except Exception as e:
    print(f"Could not connect to device. Error: {e}")
    exit()

def display_preprocessed_video():
    # Load all frame file paths, sorted correctly by filename
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_FOLDER, "*.png")))
    
    if not frame_paths:
        print(f"Error: No image files found in the folder: {FRAMES_FOLDER}. Did you transfer the folder?")
        return

    print(f"Starting playback of {len(frame_paths)} frames at a target of {TARGET_FPS} FPS. Press Ctrl+C to stop.")
    
    # Loop continuously through the frames
    while True:
        # Use a list of images to pre-load them into memory for slightly better performance.
        # This reduces file-read overhead inside the loop.
        loaded_images = []
        print("Loading all images into memory...")
        # Note: You can load the images outside of the `while True` loop if you are not clearing memory.
        # For simplicity and to match your original structure, I've left it inside.
        for frame_file in frame_paths:
            # Open and convert to 1-bit monochrome, the fastest mode.
            loaded_images.append(Image.open(frame_file).convert('1'))
        print("Image loading complete. Starting display loop.")
        
        # Start the timer for performance measurement
        start_time_total = time.monotonic()
        
        # Display the pre-loaded images with delay
        for img in loaded_images:
            # 1. Record the time *before* displaying the frame
            frame_start_time = time.monotonic()
            
            # 2. Display the frame (This is the slowest part)
            device.display(img)
            
            # 3. Record the time *after* displaying the frame
            frame_end_time = time.monotonic()
            
            # 4. Calculate how long the display took
            display_duration = frame_end_time - frame_start_time
            
            # 5. Calculate the remaining time needed for the target frame rate
            # If the display took longer than the target time, sleep_time will be negative or zero.
            sleep_time = TARGET_FRAME_TIME - display_duration
            
            # 6. Wait (sleep) if necessary
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Calculate and print actual FPS for one loop
        end_time_total = time.monotonic()
        total_duration = end_time_total - start_time_total
        
        # Avoid division by zero if total_duration is extremely small
        if total_duration > 0:
            actual_fps = len(loaded_images) / total_duration
            print(f"One loop complete: {len(loaded_images)} frames displayed in {total_duration:.2f}s. Actual FPS: {actual_fps:.2f} (Target was {TARGET_FPS}).")
        else:
             print("One loop complete, duration was too short to measure accurately.")

if __name__ == "__main__":
    try:
        display_preprocessed_video()
    except KeyboardInterrupt:
        print("\nVideo playback stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Always clear the screen when finished
        device.clear() 
        print("Screen cleared.")