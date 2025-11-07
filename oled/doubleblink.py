from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306 
from PIL import Image
import time
import os
import glob 

# --- Configuration ---
# NOTE: You MUST check the correct I2C addresses for your two OLED screens.
I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3d    
RIGHT_OLED_ADDRESS = 0x3c

# The directory containing all the image frames
FRAME_DIRECTORY = "right_frames" 

# --- NEW ZOOM CONFIGURATION ---
ZOOM_FACTOR = 0.70
# ---------------------

# --- VIDEO PLAYBACK CONFIGURATION (NEW!) ---
# 🌟 Set the desired speed here! 
# Common video FPS are 24, 30, or 60.
DESIRED_FPS = 24
# -------------------------------------------

# Calculate the required delay between frames based on the desired FPS
if DESIRED_FPS > 0:
    # Frame Delay = 1 / FPS. 
    # For example, 1 / 20 FPS = 0.05 seconds delay.
    FRAME_DELAY = 1.0 / DESIRED_FPS
else:
    # Avoid dividing by zero if FPS is set to 0
    FRAME_DELAY = 0.0 
    print("Warning: DESIRED_FPS is set to 0. Video will play as fast as possible.")


def setup_device(address, name):
    """Initializes a single SSD1306 device."""
    try:
        serial = i2c(port=I2C_PORT, address=address)
        device = ssd1306(serial)
        print(f"{name} OLED device (Address: {hex(address)}) initialized successfully.")
        return device
    except Exception as e:
        # Returning a placeholder object with clear method to prevent later errors
        class DummyDevice:
            def clear(self): pass
            def display(self, image): pass
            width = 128
            height = 64
        print(f"ERROR: Could not connect to {name} device (Address: {hex(address)}). Detail: {e}")
        return DummyDevice() 

def process_frame(img, left_device, right_device, frame_number):
    """Processes a single image frame (zoom, split, rotate) and displays it."""
    
    try:
        if img.size != (128, 128):
            print(f"Warning: Frame {frame_number} size is {img.size}. Skipping.")
            return 

        # --- Zoom Implementation ---
        target_crop_size = int(128 * ZOOM_FACTOR)
        offset = (128 - target_crop_size) // 2
        crop_box = (offset, offset, 128 - offset, 128 - offset) 
        zoomed_crop = img.crop(crop_box)
        img_zoomed = zoomed_crop.resize((128, 128), Image.BILINEAR)
        # ---------------------------

        # --- Image Splitting ---
        left_half_pil = img_zoomed.crop((0, 0, 64, 128))
        right_half_pil = img_zoomed.crop((64, 0, 128, 128))

        # --- Rotation and Conversion ---
        left_rotated = left_half_pil.rotate(-90, expand=True).rotate(180)
        left_oled_image = left_rotated.convert('1')
        
        right_rotated = right_half_pil.rotate(90, expand=True).rotate(180)
        right_oled_image = right_rotated.convert('1')

    except Exception as e:
        print(f"ERROR processing frame {frame_number}: {e}")
        return

    # 3. Send the image halves to the respective OLED displays
    try:
        left_device.display(left_oled_image)
        right_device.display(right_oled_image)
        
    except Exception as e:
        print(f"An error occurred while displaying frame {frame_number}: {e}")

def play_video():
    """Initializes devices, finds all frames, and plays them sequentially in a loop."""
    
    # 1. Setup both connections
    left_device = setup_device(LEFT_OLED_ADDRESS, "LEFT")
    right_device = setup_device(RIGHT_OLED_ADDRESS, "RIGHT")

    if isinstance(left_device, type(None)) or isinstance(right_device, type(None)):
        print("Cannot proceed without both devices connected.")
        return

    # 2. Find all frame files
    search_path = os.path.join(FRAME_DIRECTORY, "*.png")
    frame_files = sorted(glob.glob(search_path)) 
    
    if not frame_files:
        print(f"ERROR: No image files found in the directory '{FRAME_DIRECTORY}'.")
        return

    print(f"Found {len(frame_files)} frames. Starting video playback at {DESIRED_FPS} FPS...")
    print("Press Ctrl+C to stop the playback and clear the screens.")

    # 3. Loop through all frames and display them 
    # The 'while True' block creates the continuous loop!
    try:
        while True:
            for i, file_path in enumerate(frame_files):
                start_time = time.time()
                
                # Open the next frame image
                current_frame_img = Image.open(file_path)
                
                # Process and display the frame
                process_frame(current_frame_img, left_device, right_device, i)
                
                # Wait for the next frame time to maintain the framerate (FPS)
                time_spent = time.time() - start_time
                sleep_time = FRAME_DELAY - time_spent
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # else: the system is too slow to maintain the desired framerate

    except KeyboardInterrupt:
        print("\nVideo playback interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during playback: {e}")

    finally:
        # 4. Cleanup: Always clear the screens
        left_device.clear() 
        right_device.clear()
        print("Video playback finished. Screens cleared.")


if __name__ == "__main__":
    play_video()
