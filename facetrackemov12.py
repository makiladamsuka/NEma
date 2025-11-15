# =================================================================
# MODIFIED facetrackemov11.py
# This version integrates WebRTC control to override autonomous mode.
# =================================================================

import cv2
import numpy as np
import tensorflow as tf
import os
import time
from picamera2 import Picamera2
from simple_pid import PID
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import sys
import asyncio
import json
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from signal import SIGINT, SIGTERM

# --- NEW: Import the gesture module ---
import head_gestures

from oled.emodisplay import setup_and_start_display, display_emotion


# --- WebRTC/AsyncIO Setup ---
# Setup logging for the new WebRTC components
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_controller")
pc = RTCPeerConnection()

# --- CONTROL FLAGS AND VARIABLES ---
# This flag determines which control mode is active
IS_WEBRTC_CONTROL = False

# Global angle tracking (used for both autonomous and manual modes)
current_pan_angle = 90.0
current_tilt_angle = 90.0
SMOOTHING_ALPHA = 0.3  # Alpha for manual joystick smoothing (from pi.py)

# Manual Control Target (updated by the data channel)
target_manual_pan_angle = 90.0
target_manual_tilt_angle = 90.0


setup_and_start_display()
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# Servo Hardware Setup (Channels are 1 and 0 in this file, which is fine)
I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90

REQUIRED_EMOTION_FRAMES = 3
CURRENT_EMOTION = "idle"
CANDIDATE_EMOTION = "idle"
EMOTION_COUNTER = 0


PAN_Kp, PAN_Ki, PAN_Kd =  10, .001, 1
TILT_Kp, TILT_Ki, TILT_Kd = 10, .001, 1

# Note: The smoothing factors here will ONLY be used in autonomous mode.
SMOOTHING_FACTOR = .008
RETURN_SMOOTHING_FACTOR = 0.09
PID_MAX_OFFSET = 60

# Model Paths
MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media2.tflite'
YUNET_MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/face_detection_yunet_2023mar.onnx' 
YUNET_INPUT_SIZE = (320, 320) 
EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.52


# Persistence Variables
last_face_x = FRAME_WIDTH / 2
last_face_y = FRAME_HEIGHT / 2
IS_SEARCHING = False
MAX_SEARCH_FRAMES = 50
search_frame_counter = 0


# --- Servo Angle Range (Needed for manual control mapping) ---
# NOTE: These values are taken from pi.py for consistency in manual control
MANUAL_PAN_SERVO_MIN_ANGLE = 50
MANUAL_PAN_SERVO_MAX_ANGLE = 150
MANUAL_TILT_SERVO_MIN_ANGLE = 30
MANUAL_TILT_SERVO_MAX_ANGLE = 150


# --- Hardware/Model Initialization (Unchanged) ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite emotion model: {e}")
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape']
TFLITE_INPUT_H, TFLITE_INPUT_W = INPUT_SHAPE[1], INPUT_SHAPE[2]

try:
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        YUNET_INPUT_SIZE,
        0.4,
        0.3,
        5000
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}")
    sys.exit(1)


try:
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50

        pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
        tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    current_pan_angle = PAN_CENTER # Initialize global tracking variables
    current_tilt_angle = TILT_CENTER
    target_manual_pan_angle = PAN_CENTER # Initialize manual target
    target_manual_tilt_angle = TILT_CENTER

    print(f"PCA9685 initialized. Servos set to {PAN_CENTER}°, {TILT_CENTER}° tilt.")

except ValueError:
    print("Error: Could not find PCA9685 at the specified I2C address.")
    sys.exit(1)
except ImportError as e:
    print(f"Error: Required library not found ({e}). Ensure adafruit-blinka and adafruit-circuitpython-pca9685 are installed.")
    sys.exit(1)


PAN_SETPOINT = FRAME_WIDTH / 2
TILT_SETPOINT = FRAME_HEIGHT / 2

pan_pid = PID(PAN_Kp, PAN_Ki, PAN_Kd, setpoint=PAN_SETPOINT)
pan_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)

tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
tilt_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)


picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
)

picam2.configure(config)
picam2.start()
time.sleep(1.0)
print(f"Picamera2 started at {FRAME_WIDTH}x{FRAME_HEIGHT}.")


# =================================================================
# --- WEBRTC CONTROL HANDLER (New) ---
# This function is called by the data channel to set the manual target.
# It implements the mapping logic from pi.py.
# =================================================================

def set_manual_servo_targets(x, y):
    """Maps joystick data (-127 to 127) to a new target angle."""
    global target_manual_pan_angle, target_manual_tilt_angle

    # 1. CALCULATE TARGET ANGLE (Mapping from pi.py)
    # The 'y' input is inverted in the pi.py message handler: move_servos(x, -y)
    # We invert it here to match the target logic.
    y_adjusted = -y

    normalized_pan = (x + 127.0) / 254.0
    normalized_tilt = (y_adjusted + 127.0) / 254.0

    pan_range = MANUAL_PAN_SERVO_MAX_ANGLE - MANUAL_PAN_SERVO_MIN_ANGLE
    tilt_range = MANUAL_TILT_SERVO_MAX_ANGLE - MANUAL_TILT_SERVO_MIN_ANGLE

    target_manual_pan_angle = (normalized_pan * pan_range) + MANUAL_PAN_SERVO_MIN_ANGLE
    target_manual_tilt_angle = (normalized_tilt * tilt_range) + MANUAL_TILT_SERVO_MIN_ANGLE

    # 2. Clamp values immediately (though smoothing/clamping happens in the main loop too)
    target_manual_pan_angle = max(MANUAL_PAN_SERVO_MIN_ANGLE, min(MANUAL_PAN_SERVO_MAX_ANGLE, target_manual_pan_angle))
    target_manual_tilt_angle = max(MANUAL_TILT_SERVO_MIN_ANGLE, min(MANUAL_TILT_SERVO_MAX_ANGLE, target_manual_tilt_angle))

    logger.debug(f"Manual Targets set -> Pan: {target_manual_pan_angle:.1f}°, Tilt: {target_manual_tilt_angle:.1f}°")

# =================================================================
# --- WEBRTC EVENT HANDLERS (From pi.py) ---
# =================================================================

@pc.on("datachannel")
def on_datachannel(channel):
    logger.info(f"Data channel '{channel.label}' created")
    global IS_WEBRTC_CONTROL

    if channel.label != "controls":
        logger.warning(f"Ignoring data channel: {channel.label}")
        return

    @channel.on("message")
    def on_message(message):
        try:
            data = json.loads(message)
            j1_data = data.get('j1', {})
            x = j1_data.get('x', 0)
            y = j1_data.get('y', 0)

            # Update the global manual targets
            set_manual_servo_targets(x, y)

        except Exception as e:
            logger.warning(f"Error processing message: {e}")

@pc.on("icecandidate")
def on_icecandidate(candidate):
    if candidate:
        logger.info("--- NEW ROBOT ICE CANDIDATE (Copy to Web App) ---")
        print(json.dumps(candidate.to_dict()))

@pc.on("connectionstatechange")
async def on_connectionstatechange():
    global IS_WEBRTC_CONTROL, current_pan_angle, current_tilt_angle

    logger.info(f"Connection state is {pc.connectionState}")

    if pc.connectionState == "connected":
        IS_WEBRTC_CONTROL = True
        display_emotion("manual") # Indicate manual mode on the OLED
        logger.info("WebRTC Connected. **SWITCHING TO MANUAL CONTROL MODE**")
        # Set manual targets to current angle to prevent a sudden jump
        target_manual_pan_angle = current_pan_angle
        target_manual_tilt_angle = current_tilt_angle


    elif pc.connectionState in ("failed", "closed", "disconnected"):
        if IS_WEBRTC_CONTROL:
            IS_WEBRTC_CONTROL = False
            # Reset the PID controllers to prevent a large error from the
            # last manual position causing a sudden jump in autonomous mode
            pan_pid.reset()
            tilt_pid.reset()

            # Set autonomous angle targets to the current manual angle
            # This ensures a smooth transition back to the center/last-known-face.
            # We don't need to do anything with current_pan/tilt_angle as they are
            # already at the final position of the manual control.

            display_emotion("looking") # Go back to the default autonomous state
            logger.info("WebRTC Disconnected. **SWITCHING BACK TO AUTONOMOUS MODE**")


# =================================================================
# --- WEBRTC SIGNALING AND SHUTDOWN FUNCTIONS (From pi.py) ---
# =================================================================

async def run_signaling():
    """Handles the manual copy-paste signaling."""
    loop = asyncio.get_running_loop()

    logger.info("Paste Offer from Web App (single JSON object):")
    # Use loop.run_in_executor for blocking input
    offer_json = await loop.run_in_executor(None, sys.stdin.readline)
    try:
        offer = json.loads(offer_json)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer['sdp'], type=offer['type']))
        logger.info("Offer received.")
    except Exception as e:
        logger.error(f"Failed to process offer: {e}")
        return

    logger.info("Paste ICE Candidates from Web App (one JSON per line, blank line to end):")
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line.strip():
            logger.info("Finished adding Web App ICE candidates.")
            break
        try:
            candidate_json = json.loads(line)
            candidate = RTCIceCandidate(
                sdpMid=candidate_json['sdpMid'],
                sdpMLineIndex=candidate_json['sdpMLineIndex'],
                candidate=candidate_json['candidate']
            )
            await pc.addIceCandidate(candidate)
            logger.info("Added Web App ICE candidate.")
        except Exception as e:
            logger.warning(f"Invalid ICE candidate: {e}")

    # Create and send Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("--- YOUR ANSWER (Copy to Web App) ---")
    print(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))

    logger.info("Waiting for connection... (Copy any new Robot ICE candidates to the Web App)")
    # We do NOT wait indefinitely here. We return so the main loop can start.


async def shutdown(signal, loop):
    """Graceful shutdown."""
    logger.info(f"Received exit signal {signal.name}...")

    # Close peer connection
    await pc.close()

    # Stop all asyncio tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    logger.info("Cancelling tasks...")
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup hardware and exit
    cleanup_and_exit(is_async=True)


def cleanup_and_exit(is_async=False):
    """Stops the camera, resets servos, and closes OpenCV windows."""
    print("\nStopping camera and resetting servos...")
    picam2.stop()
    cv2.destroyAllWindows()
    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    time.sleep(0.5)
    if not is_async:
        sys.exit(0)


# =================================================================
# --- 5. ASYNC MAIN CONTROL LOOP ---
# The original while True loop is now an async function
# =================================================================

async def main_control_loop():
    global current_pan_angle, current_tilt_angle
    global IS_SEARCHING, HAS_DISPLAYED_SAD, search_frame_counter
    global CANDIDATE_EMOTION, CURRENT_EMOTION, EMOTION_COUNTER

    while True:
        # Give control back to the event loop frequently
        await asyncio.sleep(0.001) # Small sleep to allow WebRTC I/O to happen

        # Capture frame (must be run in executor since it's a blocking I/O)
        # Note: For simple Picamera2, sometimes it runs fine without executor,
        # but for true non-blocking, a dedicated thread is better. For this example,
        # we treat it as blocking and run it directly since it's the main bottleneck.
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to BGR

        # --- MODE SWITCHING: MANUAL OVERRIDE ---
        if IS_WEBRTC_CONTROL:
            # ----------------------------------------------------
            # 1. MANUAL CONTROL BLOCK
            # ----------------------------------------------------
            emotion_text = "Manual Control"
            emotion_color = (0, 255, 255) # Cyan for manual

            # Target is the manual target set by set_manual_servo_targets()
            target_pan_angle = target_manual_pan_angle
            target_tilt_angle = target_manual_tilt_angle

            # Apply motion smoothing from pi.py (SMOOTHING_ALPHA)
            new_pan_angle = (current_pan_angle * (1.0 - SMOOTHING_ALPHA)) + (target_pan_angle * SMOOTHING_ALPHA)
            new_tilt_angle = (current_tilt_angle * (1.0 - SMOOTHING_ALPHA)) + (target_tilt_angle * SMOOTHING_ALPHA)

            # Update the current angles
            current_pan_angle = new_pan_angle
            current_tilt_angle = new_tilt_angle

            # Keep the PIDs reset (not strictly necessary but safe)
            pan_pid.reset()
            tilt_pid.reset()

        else:
            # ----------------------------------------------------
            # 2. AUTONOMOUS (Face/Emotion Tracking) CONTROL BLOCK
            # ----------------------------------------------------

            face_detector.setInputSize((FRAME_WIDTH, FRAME_HEIGHT))
            success, faces = face_detector.detect(frame)

        target_pan_angle = PAN_CENTER
        target_tilt_angle = TILT_CENTER
        emotion_text = "Searching..."
        emotion_color = (255, 255, 255)

            if faces is not None:
                # --- Face Found (Autonomous Tracking Logic) ---
                # (The exact face-finding, group-centering, PID, and emotion logic remains unchanged)
                max_area = 0
                closest_face_coords = None
                total_x = 0
                total_y = 0
                num_faces = len(faces)

            for face in faces:
                (x_i, y_i, w_i, h_i) = map(int, face[:4])
                area = w_i * h_i
                
                # 1. Accumulate totals for the group center
                total_x += (x_i + w_i // 2)
                total_y += (y_i + h_i // 2)
                
                # 2. Find the closest face (for emotion focus)
                if area > max_area:
                    max_area = area
                    closest_face_coords = (x_i, y_i, w_i, h_i)
                
                # Draw a light green box around *every* face found
                cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 150, 0), 2) 


            # --- NEW: Calculate the two different targets ---
            
            # Target 1: The PID target (where to look)
            # This is the average center of the group
            group_center_x = total_x / num_faces
            group_center_y = total_y / num_faces
            
            # Target 2: The Emotion/Drawing target (who to focus on)
            # These are the coords of the closest face
            (x, y, w, h) = closest_face_coords
            
            # --- END NEW CALCS ---

            # We found faces, so proceed with all original logic
            IS_SEARCHING = False 
            HAS_DISPLAYED_SAD = False
            search_frame_counter = 0

            # --- Update Last Known Position (of the GROUP) ---
            last_face_x = group_center_x
            last_face_y = group_center_y

                pan_offset = pan_pid(group_center_x)
                tilt_offset = tilt_pid(group_center_y)

                # --- Emotion Detection/Gesture Logic (Unchanged) ---
                x_end = min(x + w, FRAME_WIDTH)
                y_end = min(y + h, FRAME_HEIGHT)
                x_start = max(0, x)
                y_start = max(0, y)

                roi_color = frame[y_start:y_end, x_start:x_end]

                if roi_color.size > 0:
                    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(roi_gray, (TFLITE_INPUT_W, TFLITE_INPUT_H), interpolation=cv2.INTER_AREA)
                    input_data = resized_face.astype('float32') / 255.0
                    input_data = np.expand_dims(input_data, axis=0)
                    input_data = np.expand_dims(input_data, axis=-1)

                    if input_data.shape != tuple(INPUT_SHAPE):
                        input_data = input_data.reshape(INPUT_SHAPE)

                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])

                    max_index = np.argmax(predictions[0])
                    max_confidence = predictions[0][max_index]

                    detected_emotion_this_frame = "idle"

                    if max_confidence >= CONFIDENCE_THRESHOLD:
                        predicted_emotion = EMOTION_LABELS[max_index]
                        emotion_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
                        emotion_color = (0, 255, 0)
                        detected_emotion_this_frame = predicted_emotion
                    else:
                        emotion_text = "Tracking..."
                        emotion_color = (255, 255, 0)
                        detected_emotion_this_frame = "idle"

                    if detected_emotion_this_frame == CANDIDATE_EMOTION:
                        EMOTION_COUNTER += 1
                    else:
                        EMOTION_COUNTER = 1
                        CANDIDATE_EMOTION = detected_emotion_this_frame

                    if EMOTION_COUNTER >= REQUIRED_EMOTION_FRAMES and CANDIDATE_EMOTION != CURRENT_EMOTION:
                        CURRENT_EMOTION = CANDIDATE_EMOTION
                        if CURRENT_EMOTION != "idle":
                            display_emotion(CURRENT_EMOTION)
                            gesture_to_perform = None
                            if CURRENT_EMOTION == "Happy" or CURRENT_EMOTION == "Smile":
                                gesture_to_perform = "happy_shake"

                            if gesture_to_perform:
                                print(f"--- Emotive Gesture Triggered: {gesture_to_perform} ---")
                                pan_pid.reset()
                                tilt_pid.reset()

                                # Perform the gesture (this is a blocking call, but is necessary)
                                new_pan, new_tilt = head_gestures.perform_gesture(
                                    gesture_to_perform,
                                    pan_servo,
                                    tilt_servo,
                                    current_pan_angle,
                                    current_tilt_angle
                                )

                                # Update state after gesture
                                current_pan_angle = new_pan
                                current_tilt_angle = new_tilt
                                print(f"--- Gesture Complete. Resuming PID tracking. ---")

                # --- Drawing for Face and Emotion (Unchanged) ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
                cv2.circle(frame, (int(group_center_x), int(group_center_y)), 7, (255, 0, 255), -1)
                cv2.putText(frame, "GROUP CENTER", (int(group_center_x) + 10, int(group_center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Final Autonomous PID Target Angle
                target_pan_angle = PAN_CENTER + pan_offset
                target_tilt_angle = TILT_CENTER - tilt_offset

            else:
                # --- NO FACE DETECTED (Search/Idle Logic) ---
                IS_SEARCHING = True
                display_emotion("looking")

                CURRENT_EMOTION = "idle"
                CANDIDATE_EMOTION = "idle"
                EMOTION_COUNTER = 0

                if search_frame_counter < MAX_SEARCH_FRAMES:
                    # Use Last Known Position (Momentum/Search)
                    pan_offset = pan_pid(last_face_x)
                    tilt_offset = tilt_pid(last_face_y)
                    emotion_text = f"Searching ({search_frame_counter}/{MAX_SEARCH_FRAMES})"
                    emotion_color = (255, 0, 255)
                    search_frame_counter += 1
                    cv2.circle(frame, (int(last_face_x), int(last_face_y)), 10, (255, 0, 255), 2)
                    cv2.putText(frame, "LAST POS", (int(last_face_x) + 15, int(last_face_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    target_pan_angle = PAN_CENTER + pan_offset
                    target_tilt_angle = TILT_CENTER - tilt_offset

                else:
                    # Give Up and Go to Center (Idle)
                    if not HAS_DISPLAYED_SAD:
                        display_emotion("sad")
                        pan_pid.reset()
                        tilt_pid.reset()
                        HAS_DISPLAYED_SAD = True

                        # Freeze motion by calculating offset to hold current angle
                        pan_offset = current_pan_angle - PAN_CENTER
                        tilt_offset = TILT_CENTER - current_tilt_angle
                        emotion_text = "Sad..."
                        emotion_color = (0, 0, 255)

                        target_pan_angle = PAN_CENTER + pan_offset
                        target_tilt_angle = TILT_CENTER - tilt_offset

                    else:
                        # Move to center slowly
                        pan_offset = 0
                        tilt_offset = 0
                        emotion_text = "Idle"
                        emotion_color = (128, 128, 128)
                        pan_pid.reset()
                        tilt_pid.reset()

                        target_pan_angle = PAN_CENTER + pan_offset
                        target_tilt_angle = TILT_CENTER - tilt_offset


            # --- Autonomous Servo Angle Smoothing and Calculation ---
            is_returning_to_center = (IS_SEARCHING and search_frame_counter >= MAX_SEARCH_FRAMES)

            if is_returning_to_center:
                # Faster smoothing when returning to center
                current_pan_angle = (target_pan_angle * RETURN_SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - RETURN_SMOOTHING_FACTOR))
                current_tilt_angle = (target_tilt_angle * RETURN_SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - RETURN_SMOOTHING_FACTOR))
            else:
                # Normal smoothing for tracking/searching
                current_pan_angle = (target_pan_angle * SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - SMOOTHING_FACTOR))
                current_tilt_angle = (target_tilt_angle * SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - SMOOTHING_FACTOR))


        # ----------------------------------------------------
        # 3. COMMON SERVO OUTPUT BLOCK (Always runs)
        # ----------------------------------------------------
        # The 'current_pan_angle' and 'current_tilt_angle' are now set by either the
        # Manual Control Block or the Autonomous Control Block.

        # Clamping (always apply this)
        final_pan_angle = max(0, min(180, current_pan_angle))
        final_tilt_angle = max(0, min(180, current_tilt_angle))

        current_pan_angle = final_pan_angle
        current_tilt_angle = final_tilt_angle

        pan_servo.angle = current_pan_angle
        tilt_servo.angle = current_tilt_angle


        # --- Drawing & Display (Unchanged) ---
        cv2.circle(frame, (int(PAN_SETPOINT), int(TILT_SETPOINT)), 5, (0, 0, 255), -1)
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_color, 2)

        cv2.imshow('YuNet Face Tracking & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the loop breaks (e.g., 'q' pressed), raise an exception to exit
    raise Exception("User exit")

# =================================================================
# --- MAIN EXECUTION ---
# =================================================================

async def main():
    """Starts the signaling and the main control loop concurrently."""
    await asyncio.gather(
        run_signaling(),         # Handles the WebRTC setup
        main_control_loop(),     # Handles the servo control and CV
        return_exceptions=True
    )


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # Add signal handlers for graceful shutdown
    for sig in (SIGINT, SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, loop)))

    try:
        loop.run_until_complete(main())
    except Exception as e:
        if str(e) != "User exit":
            print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        logger.info("Cleaning up...")
        # Final cleanup before process exit
        cleanup_and_exit()
        loop.close()
        logger.info("Shutdown complete.")