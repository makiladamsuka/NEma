# head_gestures.py
# This module controls "blocking" head gestures, relative to the current position.

import time
import math
from adafruit_motor import servo

# --- Private Helper Function ---

def _animate_servo(servo_obj, start_angle, end_angle, duration_sec):
    """
    Smoothly moves a servo from a start to an end angle over a duration
    using an "ease-in-out" cosine function for natural motion.
    This is a blocking function.
    """
    
    # --- Safety checks ---
    if math.isclose(start_angle, end_angle):
        return
        
    # If duration is tiny, just snap to the end
    if duration_sec < 0.01:
        servo_obj.angle = end_angle
        return

    # --- Easing Logic (Step-Based) ---
    steps = 20  # Number of small movements
    
    # Ensure sleep_time isn't impossibly small, but allow for fast moves
    sleep_time = max(0.001, duration_sec / steps)
    
    # Calculate the total change in angle
    angle_change = end_angle - start_angle

    for i in range(steps + 1):
        # 1. Calculate progress from 0.0 to 1.0
        progress = i / steps
        
        # 2. Apply the ease-in-out cosine formula
        # This formula ( (1 - cos(progress * pi)) / 2 )
        # creates a curve that starts slow, speeds up, and ends slow.
        ease_progress = (1 - math.cos(progress * math.pi)) / 2.0
        
        # 3. Calculate the new angle based on the *eased* progress
        current_angle = start_angle + (angle_change * ease_progress)
        
        # 4. Clamp and set the servo angle
        servo_obj.angle = max(0, min(180, current_angle))
        
        # 5. Wait for the correct slice of time
        # Don't sleep on the very last step
        if i < steps:
            time.sleep(sleep_time)
            
    # Ensure it lands precisely on the end_angle
    servo_obj.angle = max(0, min(180, end_angle))

# --- Gesture Functions ---

def _shake_head(pan_servo, start_angle):
    """
    Performs a "no" or "happy shake" gesture relative to the start_angle.
    Returns the final angle (which is the start_angle).
    """
    print(f"  [Gesture] Shaking head from start: {start_angle:.1f}°")
    # Increased shake amount and a realistic time for smoother motion
    shake_amount = 25  # How many degrees to move left/right
    anim_time = 0.12   # Time for one move (e.g., center to left)
    
    # Calculate target positions, clamping them to 0-180
    pos_left = max(0, start_angle - shake_amount)
    pos_right = min(180, start_angle + shake_amount)

    # --- Cycle 1 ---
    # 1. Move to left
    _animate_servo(pan_servo, start_angle, pos_left, anim_time)
    # 2. Move to right (double duration since it crosses the center)
    _animate_servo(pan_servo, pos_left, pos_right, anim_time * 2)
    # 3. Return to start
    _animate_servo(pan_servo, pos_right, start_angle, anim_time)

    # --- Cycle 2 (Re-enabled for a better shake) ---
    _animate_servo(pan_servo, start_angle, pos_left, anim_time)
    _animate_servo(pan_servo, pos_left, pos_right, anim_time * 2)
    _animate_servo(pan_servo, pos_right, start_angle, anim_time)
    
    print(f"  [Gesture] Shake complete, returning to {start_angle:.1f}°")
    return start_angle # Return to the original position

def _nod_head(tilt_servo, start_angle):
    """
    Performs a "yes" nod gesture relative to the start_angle.
    Assumes smaller angle = UP, larger angle = DOWN.
    Returns the final angle (which is the start_angle).
    """
    print(f"  [Gesture] Nodding head from start: {start_angle:.1f}°")
    nod_amount = 20    # How many degrees to move up/down
    anim_time = 0.15   # Time for one move
    
    # Remember: TILT_CENTER - offset. 
    # Smaller angle = UP, Larger angle = DOWN
    # pos_up = max(0, start_angle - nod_amount) # Not used in nod
    pos_down = min(180, start_angle + nod_amount)

    # --- Cycle 1 ---
    _animate_servo(tilt_servo, start_angle, pos_down, anim_time)
    _animate_servo(tilt_servo, pos_down, start_angle, anim_time)

    # --- Cycle 2 ---
    _animate_servo(tilt_servo, start_angle, pos_down, anim_time)
    _animate_servo(tilt_servo, pos_down, start_angle, anim_time)

    print(f"  [Gesture] Nod complete, returning to {start_angle:.1f}°")
    return start_angle # Return to the original position

# --- Public Main Function ---

def perform_gesture(gesture_name, pan_servo, tilt_servo, current_pan, current_tilt):
    """
    Main entry point to perform a gesture.
    This function will "block" (pause) the main loop until the gesture is done.
    
    It returns the final (pan, tilt) angles so the main loop can
    update its 'current_pan_angle' and 'current_tilt_angle' variables.
    """
    
    # By default, assume angles don't change
    final_pan = current_pan
    final_tilt = current_tilt

    if gesture_name == "happy_shake":
        # This gesture only moves the pan servo
        final_pan = _shake_head(pan_servo, current_pan)
        
    elif gesture_name == "happy_nod":
        # This gesture only moves the tilt servo
        final_tilt = _nod_head(tilt_servo, current_tilt)
        
    # --- Add more gestures here ---
    # elif gesture_name == "sad_droop":
    #    final_tilt = _droop_head(tilt_servo, current_tilt) # You would need to write _droop_head
        
    else:
        print(f"Warning: Unknown gesture '{gesture_name}' requested.")

    # Return the final position of both servos
    return final_pan, final_tilt