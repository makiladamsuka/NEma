import sys 
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306 
from PIL import Image
import time
import os
import glob 
import threading 
import collections # <-- NEW IMPORT for simple queue management (optional, but clean)

# --- Configuration ---
I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3d
RIGHT_OLED_ADDRESS = 0x3c 
BASE_DIRECTORY = "videos"
DEFAULT_EMOTION = "idle" 

ZOOM_FACTOR = 0.95
DESIRED_FPS = 30
FRAME_DELAY = 1.0 / DESIRED_FPS if DESIRED_FPS > 0 else 0.0


current_emotion = DEFAULT_EMOTION # The video currently being played
video_queue = collections.deque() # NEW: Holds requested emotions (e.g., ['happy'])
FRAME_CACHE = {} 
DEVICES = None 

# --- Function Definitions (setup_device, process_frame remain the same) ---

def setup_device(address, name):
    # (Keep your existing setup_device function here)
    """Initializes a single SSD1306 device."""
    try:
        serial = i2c(port=I2C_PORT, address=address)
        device = ssd1306(serial)
        print(f"{name} OLED device (Address: {hex(address)}) initialized successfully.")
        return device
    except Exception as e:
        class DummyDevice:
            def clear(self): pass
            def display(self, image): pass
            width = 128
            height = 64
        print(f"ERROR: Could not connect to {name} device (Address: {hex(address)}). Detail: {e}")
        return DummyDevice()

def process_frame(img, left_device, right_device, frame_number):
    # (Keep your existing process_frame function here, simplified for display)
    """Processes a single image frame (zoom, split, rotate) and displays it."""
    
    try:
        # --- Image Processing (Zoom, Split, Rotate, Convert) ---
        if img.size != (128, 128): return 
        target_crop_size = int(128 * ZOOM_FACTOR)
        offset = (128 - target_crop_size) // 2
        crop_box = (offset, offset, 128 - offset, 128 - offset) 
        img_zoomed = img.crop(crop_box).resize((128, 128), Image.BILINEAR)

        left_half_pil = img_zoomed.crop((0, 0, 64, 128))
        right_half_pil = img_zoomed.crop((64, 0, 128, 128))

        left_oled_image = left_half_pil.rotate(-90, expand=True).convert('1')
        right_oled_image = right_half_pil.rotate(90, expand=True).convert('1')

        # --- Display ---
        left_device.display(left_oled_image)
        right_device.display(right_oled_image)
            
    except Exception as e:
        pass


def load_emotion_frames(emotion):
    """Loads all image file paths for a specific emotion into the cache."""
    frame_path = os.path.join(BASE_DIRECTORY, emotion)
    search_path = os.path.join(frame_path, "*.png")
    frame_files = sorted(glob.glob(search_path)) 
    
    if not frame_files:
        return []
        
    print(f"Loaded {len(frame_files)} frames for '{emotion}'.")
    return frame_files

def input_thread_function():
    """
    This thread waits for user input and adds the requested emotion to the queue.
    """
    global video_queue, current_emotion
    print("\n--- Input Ready ---")
    print("Type an emotion (e.g., happy, angry) and press Enter. Idle is default.")
    
    while True:
        try:
            user_input = input(f"Current mode: '{current_emotion}'. Enter new emotion: ").strip().lower()
            
            if not user_input or user_input == DEFAULT_EMOTION:
                continue
                
            new_emotion = user_input
            
            # 1. Check if the frames for this new emotion are loaded (or load them)
            if new_emotion not in FRAME_CACHE:
                FRAME_CACHE[new_emotion] = load_emotion_frames(new_emotion)
                
            # 2. If the new emotion exists and has frames, add it to the queue
            if FRAME_CACHE.get(new_emotion):
                # Only add if it's not already in the queue or currently playing
                if new_emotion not in video_queue and new_emotion != current_emotion:
                    video_queue.append(new_emotion)
                    print(f"-> **{new_emotion.upper()}** added to queue. Will play next.")
            else:
                print(f"-> ERROR: Frames for '{new_emotion}' not found in the 'videos/{new_emotion}' folder.")
            
        except EOFError: 
            break
        except Exception:
            pass
        
def display_thread_function(left_device, right_device):
    """
    This function runs the continuous video playback loop, managing transitions.
    """
    global current_emotion, video_queue
    frame_index = 0
    
    # We will use this variable to detect when an emotion video finishes one full cycle.
    is_non_idle_video = False 

    while True:
        
        # --- 1. Check for a new video request ---
        if video_queue:
            # If there's a request, switch to it immediately
            requested = video_queue.popleft()
            if requested in FRAME_CACHE and FRAME_CACHE[requested]:
                current_emotion = requested
                frame_index = 0 # Reset frame counter to start video from the beginning
                is_non_idle_video = True
                print(f"--- PLAYING ONE-SHOT: {current_emotion.upper()} ---")

        # --- 2. Play the current emotion's frame ---
        emotion_frames = FRAME_CACHE.get(current_emotion)
        
        # Fallback to default idle if current is invalid
        if not emotion_frames:
            current_emotion = DEFAULT_EMOTION 
            emotion_frames = FRAME_CACHE.get(DEFAULT_EMOTION)
            is_non_idle_video = False # If we fall back, it's not a one-shot video
            if not emotion_frames:
                time.sleep(FRAME_DELAY)
                continue

        # Get the frame path
        file_path = emotion_frames[frame_index]
        start_time = time.time()
        
        # Process and display
        try:
            current_frame_img = Image.open(file_path)
            process_frame(current_frame_img, left_device, right_device, frame_index)
        except Exception:
            pass # Continue loop if a frame fails to load/process

        # --- 3. Manage the Loop (Crucial for one-shot) ---
        
        # Move to the next frame
        next_frame_index = (frame_index + 1)
        
        if next_frame_index >= len(emotion_frames):
            # The current video cycle is finished!
            if is_non_idle_video:
                # If we just finished playing a non-idle video, switch back to idle.
                print(f"--- Finished {current_emotion.upper()}. Switching to IDLE. ---")
                current_emotion = DEFAULT_EMOTION
                is_non_idle_video = False # Reset flag
                frame_index = 0 # Start idle video from beginning
            else:
                # If it's the idle video, just loop back to the start.
                frame_index = 0
        else:
            # Still playing the video, move to the next frame
            frame_index = next_frame_index

        # --- 4. Frame Rate Timing ---
        time_spent = time.time() - start_time
        sleep_time = FRAME_DELAY - time_spent
        if sleep_time > 0:
            time.sleep(sleep_time)


# ----------------------------------------------------------------------
# 🌟 MAIN EXECUTION BLOCK 🌟
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Setup Devices
    left_device = setup_device(LEFT_OLED_ADDRESS, "LEFT")
    right_device = setup_device(RIGHT_OLED_ADDRESS, "RIGHT")
    DEVICES = (left_device, right_device)

    if not all([DEVICES[0].display, DEVICES[1].display]):
        print("Fatal: Cannot proceed without connected devices.")
        sys.exit(1)

    # 2. Pre-load the default 'idle' emotion
    FRAME_CACHE[DEFAULT_EMOTION] = load_emotion_frames(DEFAULT_EMOTION)
    if not FRAME_CACHE.get(DEFAULT_EMOTION):
        print("FATAL: Default 'idle' frames are missing. Cannot continue.")
        sys.exit(1)

    # 3. Start the Input Thread (listener)
    input_thread = threading.Thread(target=input_thread_function, daemon=True)
    input_thread.start()
    
    # 4. Run the Display Thread in the main execution flow
    print(f"--- VIDEO PLAYBACK STARTED ({DESIRED_FPS} FPS) ---")
    try:
        # Pass the devices to the display thread function
        display_thread_function(left_device, right_device)
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    
    finally:
        # 5. Cleanup
        if DEVICES:
            left_device.clear() 
            right_device.clear()
        print("Screens cleared and script finished.")