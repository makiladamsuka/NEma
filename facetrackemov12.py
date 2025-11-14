# =================================================================
# FINAL MERGED SCRIPT: robot_main.py
# 
# This file combines:
# 1. The Autonomous Face/Emotion Tracking from facetrackemov12.py
# 2. The Manual Control Override from facetrackemov12.py
# 3. The Automatic Firebase/WebRTC Signaling from pi4.py
# =================================================================

# --- Core CV/Async/System Imports ---
import cv2
import numpy as np
import tensorflow as tf
import os
import time
from picamera2 import Picamera2
from simple_pid import PID
import sys
import asyncio
import json
import logging
from signal import SIGINT, SIGTERM

# --- WebRTC Imports ---
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

# --- Firebase Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- Hardware Imports ---
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import head_gestures # Your head_gestures.py
from oled.emodisplay import setup_and_start_display, display_emotion

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_controller")

# =================================================================
# --- 1. FIREBASE & WEBRTC CONFIGURATION (from pi4.py) ---
# =================================================================

# CRITICAL: REPLACE with the correct path to your Firebase service account key
FIREBASE_KEY_PATH = os.path.expanduser('/home/nema/Documents/NEma/firebase/serviceAccountKey.json') 
ROOM_ID = 'esp32_robot_room' 
CALLS_COLLECTION = 'robot_calls'

# Global variables for async loop and PeerConnection
MAIN_EVENT_LOOP = None
pc = None  # PeerConnection will be created *per call*
OFFER_PROCESSED = False

# --- Firebase Setup ---
try:
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(f"Firebase key not found at {FIREBASE_KEY_PATH}")
        
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin Initialized.")
except FileNotFoundError as e:
    logger.critical(f"FATAL ERROR: {e}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"FATAL ERROR initializing Firebase: {e}")
    sys.exit(1)

db = firestore.client()
call_doc_ref = db.collection(CALLS_COLLECTION).document(ROOM_ID)


# =================================================================
# --- 2. ROBOT HARDWARE & CV CONFIG (from facetrackemov12.py) ---
# =================================================================

# --- Control Flags ---
IS_WEBRTC_CONTROL = False # This is the "switch"

# --- Global Angle Tracking ---
current_pan_angle = 90.0
current_tilt_angle = 90.0
target_manual_pan_angle = 90.0
target_manual_tilt_angle = 90.0

# --- CV/PID Constants ---
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
PAN_SETPOINT = FRAME_WIDTH / 2
TILT_SETPOINT = FRAME_HEIGHT / 2

# --- Hardware Constants ---
I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90

# Servo Angle Range (for manual control mapping)
MANUAL_PAN_SERVO_MIN_ANGLE = 50
MANUAL_PAN_SERVO_MAX_ANGLE = 150
MANUAL_TILT_SERVO_MIN_ANGLE = 30
MANUAL_TILT_SERVO_MAX_ANGLE = 150
MANUAL_SMOOTHING_ALPHA = 0.3  # Smoothing for manual joystick control

# --- PID & Autonomous Constants ---
PAN_Kp, PAN_Ki, PAN_Kd =  10, .001, 1
TILT_Kp, TILT_Ki, TILT_Kd = 10, .001, 1
AUTONOMOUS_SMOOTHING_FACTOR = .008
RETURN_SMOOTHING_FACTOR = 0.09
PID_MAX_OFFSET = 60

# --- Emotion Constants ---
REQUIRED_EMOTION_FRAMES = 3
CURRENT_EMOTION = "idle"
CANDIDATE_EMOTION = "idle"
EMOTION_COUNTER = 0

# --- Model Paths ---
MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media2.tflite'
YUNET_MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/face_detection_yunet_2023mar.onnx'
YUNET_INPUT_SIZE = (320, 320)
EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.52

# --- State Variables ---
last_face_x = FRAME_WIDTH / 2
last_face_y = FRAME_HEIGHT / 2
IS_SEARCHING = False
HAS_DISPLAYED_SAD = False
MAX_SEARCH_FRAMES = 50
search_frame_counter = 0

# --- Global Hardware/Model Objects ---
interpreter = None
input_details = None
output_details = None
TFLITE_INPUT_H = 0
TFLITE_INPUT_W = 0
face_detector = None
pca = None
pan_servo = None
tilt_servo = None
pan_pid = None
tilt_pid = None
picam2 = None


# =================================================================
# --- 3. INITIALIZATION FUNCTIONS ---
# =================================================================

def initialize_models():
    """Loads TFLite and YuNet models."""
    global interpreter, input_details, output_details, TFLITE_INPUT_H, TFLITE_INPUT_W
    global face_detector
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        INPUT_SHAPE = input_details[0]['shape']
        TFLITE_INPUT_H, TFLITE_INPUT_W = INPUT_SHAPE[1], INPUT_SHAPE[2]
        logger.info("TFLite Emotion Model loaded.")
    except Exception as e:
        logger.critical(f"Error loading TFLite emotion model: {e}")
        return False

    try:
        face_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", YUNET_INPUT_SIZE, 0.4, 0.3, 5000
        )
        logger.info("YuNet Face Detector loaded.")
    except Exception as e:
        logger.critical(f"Error loading YuNet model: {e}")
        return False
        
    return True

def initialize_hardware():
    """Initializes PCA9685, Servos, PID controllers, and Camera."""
    global pca, pan_servo, tilt_servo, current_pan_angle, current_tilt_angle
    global pan_pid, tilt_pid, picam2
    
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=I2C_ADDRESS)
        pca.frequency = 50

        pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
        tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)

        pan_servo.angle = PAN_CENTER
        tilt_servo.angle = TILT_CENTER
        current_pan_angle = PAN_CENTER
        current_tilt_angle = TILT_CENTER
        logger.info(f"PCA9685 initialized. Servos set to {PAN_CENTER}°, {TILT_CENTER}°.")
    except Exception as e:
        logger.critical(f"Error initializing PCA9685: {e}")
        return False

    # Init PIDs
    pan_pid = PID(PAN_Kp, PAN_Ki, PAN_Kd, setpoint=PAN_SETPOINT)
    pan_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)
    tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
    tilt_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)
    logger.info("PID Controllers initialized.")

    # Init Camera
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)
        logger.info(f"Picamera2 started at {FRAME_WIDTH}x{FRAME_HEIGHT}.")
    except Exception as e:
        logger.critical(f"Error initializing Picamera2: {e}")
        return False

    # Init Display
    setup_and_start_display()
    logger.info("OLED Display thread started.")
    
    return True

def cleanup_and_exit():
    """Stops the camera, resets servos, and closes OpenCV windows."""
    logger.warning("\nStopping camera and resetting servos...")
    if picam2:
        picam2.stop()
    cv2.destroyAllWindows()
    if pan_servo and tilt_servo:
        pan_servo.angle = PAN_CENTER
        tilt_servo.angle = TILT_CENTER
    time.sleep(0.5)
    logger.info("Cleanup complete. Exiting.")


# =================================================================
# --- 4. WEBRTC SIGNALING (from pi4.py, with modifications) ---
# =================================================================

async def cleanup_firebase_room():
    """Deletes the signaling room in Firestore after disconnection."""
    logger.info("Cleaning up Firebase room...")
    try:
        await asyncio.to_thread(call_doc_ref.delete)
        logger.info("Room document deleted. Ready for next browser connection.")
    except Exception as e:
        logger.error(f"Error during Firebase cleanup: {e}")

def handle_snapshot_in_thread(col_snapshot):
    """Callback from Firebase thread, safely schedules async processing."""
    global MAIN_EVENT_LOOP
    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(
            process_offer_snapshot(col_snapshot),
            MAIN_EVENT_LOOP
        )

def handle_ice_snapshot_in_thread(pc_instance, col_snapshot):
    """Callback for ICE candidates, safely scheduled."""
    global MAIN_EVENT_LOOP
    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(
            add_ice_candidates(pc_instance, col_snapshot),
            MAIN_EVENT_LOOP
        )

async def add_ice_candidates(pc_instance, col_snapshot):
    """Adds incoming ICE candidates to the PeerConnection."""
    for change in col_snapshot:
        if change.type.name == 'ADDED':
            candidate_data = change.document.to_dict()
            try:
                if candidate_data.get('candidate'):
                    candidate = RTCIceCandidate(**candidate_data)
                    await pc_instance.addIceCandidate(candidate)
            except Exception as e:
                logger.warning(f"Error adding ICE candidate: {e}")

def set_manual_servo_targets(x, y):
    """Maps joystick data (-127 to 127) to a new target angle."""
    global target_manual_pan_angle, target_manual_tilt_angle

    # 1. CALCULATE TARGET ANGLE
    normalized_pan = (x + 127.0) / 254.0
    normalized_tilt = (y + 127.0) / 254.0 # Invert Y? You can do it here if needed.

    pan_range = MANUAL_PAN_SERVO_MAX_ANGLE - MANUAL_PAN_SERVO_MIN_ANGLE
    tilt_range = MANUAL_TILT_SERVO_MAX_ANGLE - MANUAL_TILT_SERVO_MIN_ANGLE

    # Note: Your pi4.py inverts Y. Your facetrack inverts it differently.
    # Let's use the logic from pi4.py which maps Y directly.
    # If your tilt is backward, swap MANUAL_TILT_SERVO_MAX/MIN_ANGLE in the line below.
    target_manual_pan_angle = (normalized_pan * pan_range) + MANUAL_PAN_SERVO_MIN_ANGLE
    target_manual_tilt_angle = (normalized_tilt * tilt_range) + MANUAL_TILT_SERVO_MIN_ANGLE
    
    # You might need to invert tilt like this, depending on servo mounting:
    # target_manual_tilt_angle = MANUAL_TILT_SERVO_MAX_ANGLE - (normalized_tilt * tilt_range)
    
    logger.debug(f"Manual Targets set -> Pan: {target_manual_pan_angle:.1f}°, Tilt: {target_manual_tilt_angle:.1f}°")

async def process_offer_snapshot(col_snapshot):
    """
    The main signaling function. Creates the PC and wires up all event
    handlers for THIS specific call.
    """
    global OFFER_PROCESSED, IS_WEBRTC_CONTROL, pc
    global target_manual_pan_angle, target_manual_tilt_angle, current_pan_angle, current_tilt_angle

    # Check for offer data
    try:
        doc = col_snapshot[0]
        data = doc.to_dict()
    except IndexError:
        return 

    if 'offer' in data and not OFFER_PROCESSED:
        OFFER_PROCESSED = True # Mark as processing
        logger.info("Offer received! Processing...")
        
        offer = RTCSessionDescription(
            sdp=data['offer']['sdp'],
            type=data['offer']['type'])

        # Create a new PeerConnection for every new call
        pc = RTCPeerConnection() 
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel '{channel.label}' created")

            @channel.on("message")
            def on_message(message):
                # This logic is from facetrackemov12.py
                try:
                    data = json.loads(message)
                    j1_data = data.get('j1', {})
                    x = j1_data.get('x', 0)
                    y = j1_data.get('y', 0)
                    set_manual_servo_targets(x, y) # Update the manual targets
                except Exception as e:
                    logger.warning(f"Error processing message: {e}")

            @channel.on("open")
            def on_open():
                # This logic is from facetrackemov12.py's on_connectionstatechange
                global IS_WEBRTC_CONTROL, target_manual_pan_angle, target_manual_tilt_angle
                
                IS_WEBRTC_CONTROL = True
                display_emotion("manual")
                logger.warning("WebRTC Connected. **SWITCHING TO MANUAL CONTROL MODE**")
                # Set manual targets to current angle to prevent a sudden jump
                target_manual_pan_angle = current_pan_angle
                target_manual_tilt_angle = current_tilt_angle

            @channel.on("close")
            def on_close():
                # This logic combines pi4.py and facetrackemov12.py
                global IS_WEBRTC_CONTROL, OFFER_PROCESSED
                
                logger.warning("WebRTC DataChannel CLOSED.")
                
                # 1. Reset servos to center (from pi4.py)
                pan_servo.angle = PAN_CENTER
                tilt_servo.angle = TILT_CENTER
                
                # 2. Switch back to autonomous mode (from facetrackemov12.py)
                if IS_WEBRTC_CONTROL:
                    IS_WEBRTC_CONTROL = False
                    pan_pid.reset()
                    tilt_pid.reset()
                    display_emotion("looking")
                    logger.warning("WebRTC Disconnected. **SWITCHING BACK TO AUTONOMOUS MODE**")
                
                # 3. Reset flags to allow a new incoming call
                OFFER_PROCESSED = False
                
                # 4. Schedule ASYNC cleanup (Thread-safe)
                if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(
                        cleanup_firebase_room(),
                        MAIN_EVENT_LOOP
                    )

        # --- ICE Candidate Handling (from pi4.py) ---
        offer_candidates_ref = call_doc_ref.collection('offerCandidates')
        offer_candidates_ref.on_snapshot(lambda col_snapshot, changes, read_time: 
            handle_ice_snapshot_in_thread(pc, col_snapshot))

        answer_candidates_ref = call_doc_ref.collection('answerCandidates')
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                answer_candidates_ref.add(candidate.toJSON())

        # 3. Create and Send Answer (from pi4.py)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        answer_data = {
            'answer': {
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }
        }
        await asyncio.to_thread(call_doc_ref.update, answer_data)
        logger.info("Answer sent to Firestore. Connection process initiated.")


# =================================================================
# --- 5. ASYNC MAIN CONTROL LOOP (from facetrackemov12.py) ---
# =================================================================

async def main_control_loop():
    """
    This is the main robot loop. It runs CV and servo control.
    It checks the 'IS_WEBRTC_CONTROL' flag every frame to decide
    which logic (autonomous or manual) to run.
    """
    global current_pan_angle, current_tilt_angle
    global IS_SEARCHING, HAS_DISPLAYED_SAD, search_frame_counter
    global CANDIDATE_EMOTION, CURRENT_EMOTION, EMOTION_COUNTER
    global last_face_x, last_face_y

    logger.info("Main control loop started. Waiting for tasks...")
    
    while True:
        # Give control back to the event loop frequently
        await asyncio.sleep(0.001) # Allows WebRTC I/O to happen

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to BGR

        # ---!!! THE BIG SWITCH !!!---
        if IS_WEBRTC_CONTROL:
            # ----------------------------------------------------
            # 1. MANUAL CONTROL BLOCK
            # ----------------------------------------------------
            emotion_text = "Manual Control"
            emotion_color = (0, 255, 255) # Cyan for manual

            # Target is the manual target set by set_manual_servo_targets()
            target_pan_angle = target_manual_pan_angle
            target_tilt_angle = target_manual_tilt_angle

            # Apply motion smoothing
            new_pan_angle = (current_pan_angle * (1.0 - MANUAL_SMOOTHING_ALPHA)) + (target_pan_angle * MANUAL_SMOOTHING_ALPHA)
            new_tilt_angle = (current_tilt_angle * (1.0 - MANUAL_SMOOTHING_ALPHA)) + (target_tilt_angle * MANUAL_SMOOTHING_ALPHA)

            current_pan_angle = new_pan_angle
            current_tilt_angle = new_tilt_angle

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
                max_area = 0
                closest_face_coords = None
                total_x = 0
                total_y = 0
                num_faces = len(faces)

                for face in faces:
                    (x_i, y_i, w_i, h_i) = map(int, face[:4])
                    area = w_i * h_i
                    total_x += (x_i + w_i // 2)
                    total_y += (y_i + h_i // 2)
                    if area > max_area:
                        max_area = area
                        closest_face_coords = (x_i, y_i, w_i, h_i)
                    cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 150, 0), 2)

                group_center_x = total_x / num_faces
                group_center_y = total_y / num_faces
                (x, y, w, h) = closest_face_coords

                IS_SEARCHING = False
                HAS_DISPLAYED_SAD = False
                search_frame_counter = 0

                last_face_x = group_center_x
                last_face_y = group_center_y

                pan_offset = pan_pid(group_center_x)
                tilt_offset = tilt_pid(group_center_y)

                # --- Emotion Detection/Gesture Logic ---
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

                    if input_data.shape != tuple(input_details[0]['shape']):
                        input_data = input_data.reshape(input_details[0]['shape'])

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
                                logger.info(f"--- Emotive Gesture Triggered: {gesture_to_perform} ---")
                                pan_pid.reset()
                                tilt_pid.reset()
                                new_pan, new_tilt = head_gestures.perform_gesture(
                                    gesture_to_perform, pan_servo, tilt_servo,
                                    current_pan_angle, current_tilt_angle
                                )
                                current_pan_angle = new_pan
                                current_tilt_angle = new_tilt
                                logger.info("--- Gesture Complete. Resuming PID tracking. ---")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
                cv2.circle(frame, (int(group_center_x), int(group_center_y)), 7, (255, 0, 255), -1)

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
                    pan_offset = pan_pid(last_face_x)
                    tilt_offset = tilt_pid(last_face_y)
                    emotion_text = f"Searching ({search_frame_counter}/{MAX_SEARCH_FRAMES})"
                    emotion_color = (255, 0, 255)
                    search_frame_counter += 1
                    target_pan_angle = PAN_CENTER + pan_offset
                    target_tilt_angle = TILT_CENTER - tilt_offset
                else:
                    if not HAS_DISPLAYED_SAD:
                        display_emotion("sad")
                        pan_pid.reset()
                        tilt_pid.reset()
                        HAS_DISPLAYED_SAD = True
                    emotion_text = "Idle"
                    emotion_color = (128, 128, 128)
                    pan_pid.reset()
                    tilt_pid.reset()
                    target_pan_angle = PAN_CENTER
                    target_tilt_angle = TILT_CENTER

            # --- Autonomous Servo Angle Smoothing and Calculation ---
            is_returning_to_center = (IS_SEARCHING and search_frame_counter >= MAX_SEARCH_FRAMES)
            smoothing = RETURN_SMOOTHING_FACTOR if is_returning_to_center else AUTONOMOUS_SMOOTHING_FACTOR
            current_pan_angle = (target_pan_angle * smoothing) + (current_pan_angle * (1.0 - smoothing))
            current_tilt_angle = (target_tilt_angle * smoothing) + (current_tilt_angle * (1.0 - smoothing))


        # ----------------------------------------------------
        # 3. COMMON SERVO OUTPUT BLOCK (Always runs)
        # ----------------------------------------------------
        final_pan_angle = max(0, min(180, current_pan_angle))
        final_tilt_angle = max(0, min(180, current_tilt_angle))

        current_pan_angle = final_pan_angle
        current_tilt_angle = final_tilt_angle

        pan_servo.angle = current_pan_angle
        tilt_servo.angle = current_tilt_angle

        # --- Drawing & Display ---
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_color, 2)
        cv2.imshow('YuNet Face Tracking & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the loop breaks (e.g., 'q' pressed), signal for shutdown
    logger.info("'q' pressed. Shutting down main loop.")
    raise Exception("User exit")


# =================================================================
# --- 6. MAIN EXECUTION ---
# =================================================================

async def firebase_listener():
    """A long-running task that just listens to Firebase."""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = asyncio.get_running_loop() 

    # Start watching for changes in the room document (new Offer)
    call_doc_ref.on_snapshot(lambda col_snapshot, changes, read_time: 
                             handle_snapshot_in_thread(col_snapshot))
    logger.info("Firebase listener started. Listening for new WebRTC Offers...")
    
    # Keep this task alive indefinitely
    while True:
        await asyncio.sleep(3600)

async def shutdown(signal, loop):
    """Graceful shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    
    # Close peer connection if it exists
    if pc and pc.connectionState != "closed":
        await pc.close()

    # Stop all asyncio tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    logger.info("Cancelling tasks...")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop the asyncio loop
    loop.stop()

async def main():
    """Initializes hardware and runs the main loops concurrently."""
    if not initialize_models():
        return
    if not initialize_hardware():
        return
        
    try:
        await asyncio.gather(
            firebase_listener(),   # Task 1: Listens for new calls
            main_control_loop(),   # Task 2: Runs the CV/Robot logic
            return_exceptions=True
        )
    except Exception as e:
        if "User exit" not in str(e):
            logger.error(f"An unexpected error occurred in main: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # Add signal handlers for graceful shutdown (Ctrl+C)
    for sig in (SIGINT, SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, loop)))

    try:
        loop.run_until_complete(main())
    finally:
        logger.info("Main loop exited. Cleaning up...")
        cleanup_and_exit()
        loop.close()
        logger.info("Shutdown complete.")