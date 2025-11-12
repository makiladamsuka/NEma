import time

# --- Private "Puppeteer" Variables ---
_pan_servo = None
_tilt_servo = None

_current_sequence_name = None
_current_step = 0
_last_step_time = 0.0

# --- Define Your "Animations" Here ---
# Each step is: (pan_angle, tilt_angle, time_to_wait_after_moving)
SEQUENCES = {
    "happy": [
        (90, 60, 0.15), # Look up-right
        (50, 50, 0.15),  # Look up-left
        (80, 40, 0.15), # Look up-right
        (40, 50, 0.15),  # Look up-left
        (90, 90, 0.1)    # Center
    ],
    "smile": [
        (90, 80, 0.2),   # Look up slightly
        (100, 90, 0.2),  # Tilt head right
        (80, 90, 0.2),   # Tilt head left
        (90, 90, 0.1)    # Center
    ],
    "sad": [
        (90, 120, 0.7),  # Look all the way down
        (90, 110, 0.5),  # Look up a little
        (90, 120, 0.7),  # Look down again
        (90, 90, 0.2)    # Center
    ],
    "looking": [
        (70, 90, 0.4),   # Look left
        (110, 90, 0.4),  # Look right
        (90, 90, 0.2)    # Center
    ]
    # Add more moves here! e.g., "confused", "curious"
}

# --- Public Functions (for your main script) ---

def init(pan_servo_obj, tilt_servo_obj):
    """
    Passes the servo objects from the main script to this "puppeteer".
    """
    global _pan_servo, _tilt_servo
    _pan_servo = pan_servo_obj
    _tilt_servo = tilt_servo_obj
    print("Head Movements puppeteer initialized.")

def start_move(name):
    """
    Starts a new head movement sequence.
    """
    global _current_sequence_name, _current_step, _last_step_time
    
    # Don't interrupt a move that's already playing
    if is_active():
        return
        
    if name in SEQUENCES:
        _current_sequence_name = name
        _current_step = 0
        _last_step_time = time.time()
        
        # --- Immediately perform the first step ---
        first_keyframe = SEQUENCES[_current_sequence_name][_current_step]
        if _pan_servo:
            _pan_servo.angle = first_keyframe[0]
        if _tilt_servo:
            _tilt_servo.angle = first_keyframe[1]
        print(f"Starting head move: {name}")

def is_active():
    """
    Lets the main script know if a sequence is currently playing.
    """
    return _current_sequence_name is not None

def update():
    """
    This function MUST be called every single frame in the main loop.
    It checks if it's time to move to the next step in the animation.
    """
    global _current_sequence_name, _current_step, _last_step_time
    
    if not is_active():
        return

    # Get the current step's data
    sequence = SEQUENCES[_current_sequence_name]
    current_keyframe = sequence[_current_step]
    duration_to_wait = current_keyframe[2]
    
    # Check if enough time has passed
    if (time.time() - _last_step_time) > duration_to_wait:
        # It's time to move to the next step
        _current_step += 1
        
        # Check if the sequence is over
        if _current_step >= len(sequence):
            _current_sequence_name = None # Stop
            return
            
        # Perform the next step
        next_keyframe = sequence[_current_step]
        if _pan_servo:
            _pan_servo.angle = next_keyframe[0]
        if _tilt_servo:
            _tilt_servo.angle = next_keyframe[1]
        
        # Reset the timer
        _last_step_time = time.time()