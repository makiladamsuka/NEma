import sys 
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306 
from PIL import Image
import time
import os
import glob 
import threading 
import collections 


I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3d
RIGHT_OLED_ADDRESS = 0x3c 


BASE_DIRECTORY = "/home/nema/Documents/NEma/oled/videos" 
DEFAULT_EMOTION = "idle" 

ZOOM_FACTOR = 0.95
DESIRED_FPS = 30
FRAME_DELAY = 1.0 / DESIRED_FPS if DESIRED_FPS > 0 else 0.0


current_emotion = DEFAULT_EMOTION 
video_queue = collections.deque() 
FRAME_CACHE = {} 
DEVICES = None # Stores (left_device, right_device)
DISPLAY_RUNNING = False



def _setup_device(address, name):
    """Initializes a single SSD1306 device or returns a dummy device on failure."""
    try:
        # Avoid jargon: i2c is just the way the screen talks to the computer.
        serial = i2c(port=I2C_PORT, address=address)
        device = ssd1306(serial)
        print(f"{name} OLED device (Address: {hex(address)}) initialized successfully.")
        return device
    except Exception as e:
        # Analogy: This is like a stand-in device, so the program doesn't crash.
        class DummyDevice:
            def clear(self): pass
            def display(self, image): pass
            width = 128
            height = 64
        print(f"ERROR: Could not connect to {name} device (Address: {hex(address)}). Detail: {e}")
        return DummyDevice()

def _load_emotion_frames(emotion):
    """Loads all image file paths for a specific emotion into the cache."""
    frame_path = os.path.join(BASE_DIRECTORY, emotion)
    search_path = os.path.join(frame_path, "*.png")
    frame_files = sorted(glob.glob(search_path)) 
    
    if not frame_files:
        return []
        
    print(f"Loaded {len(frame_files)} frames for '{emotion}'.")
    return frame_files

def _process_frame(img, frame_number):
    """Processes a single image frame (zoom, split, rotate) and displays it."""
    global DEVICES
    if not DEVICES: return
    left_device, right_device = DEVICES
    
    try:
        # --- Image Processing (Zoom, Split, Rotate, Convert) ---
        if img.size != (128, 128): return 
        target_crop_size = int(128 * ZOOM_FACTOR)
        offset = (128 - target_crop_size) // 2
        crop_box = (offset, offset, 128 - offset, 128 - offset) 
        img_zoomed = img.crop(crop_box).resize((128, 128), Image.BILINEAR)

        left_half_pil = img_zoomed.crop((0, 0, 64, 128))
        right_half_pil = img_zoomed.crop((64, 0, 128, 128))

        # Rotate and convert to 1-bit black/white image
        left_oled_image = left_half_pil.rotate(-90, expand=True).convert('1')
        right_oled_image = right_half_pil.rotate(90, expand=True).convert('1')

        # --- Display ---
        left_device.display(left_oled_image)
        right_device.display(right_oled_image)
            
    except Exception:
        pass

def _play_emotion_one_shot(emotion_name):
    """Plays a single full cycle of the specified emotion video."""
    
    emotion_frames = FRAME_CACHE.get(emotion_name)
    
    if not emotion_frames:
        print(f"ERROR: Cannot play '{emotion_name}'. Frames missing in cache.")
        return False
        
    print(f"--- PLAYING ONE-SHOT: {emotion_name.upper()} ({len(emotion_frames)} frames) ---")
    
    for frame_index in range(len(emotion_frames)):
        
        # Check if the program has been stopped
        if not DISPLAY_RUNNING:
            return

        file_path = emotion_frames[frame_index]
        start_time = time.time()
        
        # Process and display
        try:
            current_frame_img = Image.open(file_path)
            _process_frame(current_frame_img, frame_index) 
        except Exception:
            pass

        # Frame Rate Timing
        time_spent = time.time() - start_time
        sleep_time = FRAME_DELAY - time_spent
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"--- Finished one cycle of {emotion_name.upper()}. ---")
    return True


# --- Thread Functions ---

def _input_thread_function():
    """Input thread for command-line testing."""
    global video_queue, current_emotion, DISPLAY_RUNNING
    print("\n--- Input Ready ---")
    print("Type an emotion (e.g., happy, angry) and press Enter. 'quit' to stop.")
    
    while DISPLAY_RUNNING:
        try:
            user_input = input(f"Current mode: '{current_emotion}'. Enter new emotion: ").strip().lower()
            
            if user_input == 'quit':
                stop_display()
                break
                
            if not user_input or user_input == DEFAULT_EMOTION:
                continue
                
            display_emotion(user_input) # Use the public function
            
        except EOFError: 
            break
        except Exception:
            pass
            
def _display_thread_function():
    """The continuous video playback loop."""
    global current_emotion, video_queue, DISPLAY_RUNNING
    
    current_emotion = DEFAULT_EMOTION 
    idle_frame_index = 0 

    while DISPLAY_RUNNING:
        
        # 1. Check for a new video request (One-Shot Mode)
        if video_queue:
            requested_emotion = video_queue.popleft()
            
            _play_emotion_one_shot(requested_emotion)
            
            # After one-shot, always return to IDLE mode
            current_emotion = DEFAULT_EMOTION
            idle_frame_index = 0 

        # 2. Play the current emotion's frame (Continuous IDLE Mode)
        
        emotion_frames = FRAME_CACHE.get(DEFAULT_EMOTION)
        
        if not emotion_frames:
            time.sleep(FRAME_DELAY)
            continue

        file_path = emotion_frames[idle_frame_index]
        start_time = time.time()
        
        # Process and display the IDLE frame
        try:
            current_frame_img = Image.open(file_path)
            _process_frame(current_frame_img, idle_frame_index)
        except Exception:
            pass 

        # 3. Manage the IDLE Loop
        # Analogy: This is like reaching the end of a video clip and restarting it.
        idle_frame_index = (idle_frame_index + 1) % len(emotion_frames)
        
        # 4. Frame Rate Timing
        time_spent = time.time() - start_time
        sleep_time = FRAME_DELAY - time_spent
        if sleep_time > 0:
            time.sleep(sleep_time)



def setup_and_start_display(enable_input_thread=False):
    """
    Initializes devices, pre-loads the default emotion, and starts the continuous display loop.
    """
    global DEVICES, FRAME_CACHE, DISPLAY_RUNNING
    
    # 1. Setup Devices
    left_device = _setup_device(LEFT_OLED_ADDRESS, "LEFT")
    right_device = _setup_device(RIGHT_OLED_ADDRESS, "RIGHT")
    DEVICES = (left_device, right_device)
    
    if not all([hasattr(DEVICES[0], 'display'), hasattr(DEVICES[1], 'display')]):
        print("Fatal: Cannot proceed without device objects.")
        sys.exit(1)
        
    # 2. Pre-load the default 'idle' emotion
    FRAME_CACHE[DEFAULT_EMOTION] = _load_emotion_frames(DEFAULT_EMOTION)
    if not FRAME_CACHE.get(DEFAULT_EMOTION):
        print("FATAL: Default 'idle' frames are missing. Cannot continue.")
        sys.exit(1)

    # 3. Start the Display Thread
    DISPLAY_RUNNING = True
    display_thread = threading.Thread(target=_display_thread_function, daemon=True)
    display_thread.start()
    print(f"--- VIDEO PLAYBACK STARTED ({DESIRED_FPS} FPS) ---")
    
    # 4. Start the optional Input Thread
    if enable_input_thread:
        input_thread = threading.Thread(target=_input_thread_function, daemon=True)
        input_thread.start()
        
    return display_thread


def display_emotion(emotion_name):
    global video_queue, current_emotion, FRAME_CACHE, DISPLAY_RUNNING
    
    if not DISPLAY_RUNNING:
        print("Error: Display loop is not running. Call setup_and_start_display() first.")
        return False
        
    requested_emotion = emotion_name.strip().lower()

    # 1. Check or load frames
    if requested_emotion not in FRAME_CACHE:
        FRAME_CACHE[requested_emotion] = _load_emotion_frames(requested_emotion)
        
    # 2. Add to queue if valid
    if FRAME_CACHE.get(requested_emotion):
        # Only add if it's not currently playing or already queued
        if requested_emotion not in video_queue and requested_emotion != current_emotion:
            video_queue.append(requested_emotion)
            print(f"-> **{requested_emotion.upper()}** added to queue. Will play next.")
            # Set current_emotion immediately so the input thread sees the change
            current_emotion = requested_emotion 
            return True
        elif requested_emotion == current_emotion:
            print(f"-> **{requested_emotion.upper()}** is already playing (or about to start).")
            return True
        else:
            print(f"-> **{requested_emotion.upper()}** is already in the queue.")
            return True
    else:
        print(f"-> ERROR: Frames for '{requested_emotion}' not found in the 'videos/{requested_emotion}' folder.")
        return False

def stop_display():
    """Stops the display thread and clears the screens."""
    global DISPLAY_RUNNING, DEVICES
    print("\nStopping program...")
    DISPLAY_RUNNING = False
    
    # Give a moment for threads to notice the flag change
    time.sleep(0.5) 

    # Cleanup
    if DEVICES:
        DEVICES[0].clear() 
        DEVICES[1].clear()
    print("Screens cleared and script finished.")

# --- Example Usage (When run as main script) ---
if __name__ == "__main__":
    
    # Start the display loop and the interactive input thread
    # Returns the thread object, but the daemon thread runs automatically
    display_thread = setup_and_start_display(enable_input_thread=True) 

    # Keep the main thread alive so the daemon threads can run
    try:
        # Keep the main thread alive until the display loop is told to stop
        while DISPLAY_RUNNING:
            time.sleep(1) 
            
    except KeyboardInterrupt:
        stop_display()
    
