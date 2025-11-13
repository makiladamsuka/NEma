import asyncio
import json
import logging
import sys
import os

# --- NEW FIREBASE IMPORTS ---
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.firestore import client as FirestoreClient
# --- END FIREBASE IMPORTS ---

from adafruit_servokit import ServoKit
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from signal import SIGINT, SIGTERM

# --- CONFIGURATION ---
PAN_SERVO_GPIO = 1  # Pan (left/right) servo, connected to channel 0
TILT_SERVO_GPIO = 0 # Tilt (up/down) servo, connected to channel 1
PAN_SERVO_MIN_ANGLE = 50
PAN_SERVO_MAX_ANGLE = 150
TILT_SERVO_MIN_ANGLE = 30
TILT_SERVO_MAX_ANGLE = 150
SMOOTHING_ALPHA = 0.3
ROOM_ID = "robot-101" # MUST match the ID entered on the web app!
# --- END CONFIGURATION ---

current_pan_angle = 90.0
current_tilt_angle = 90.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_pi")

# --- FIREBASE SETUP ---
db = None
try:
    # IMPORTANT: Change "serviceAccountKey.json" to your file name if different.
    cred = credentials.Certificate("piwebrtc-firebase-adminsdk-fbsvc-8d7f914920.json")
    firebase_admin.initialize_app(cred)
    db: FirestoreClient = firestore.client()
    logger.info("Firebase Admin SDK initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
    
# --- SERVO INITIALIZATION (SAME AS BEFORE) ---
# ... (Servos initialization block from original pi.py)
# This block remains unchanged for ServoKit setup and error handling
kit = None
servos_enabled = False
try:
    kit = ServoKit(channels=16) 
    kit.servo[PAN_SERVO_GPIO].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    kit.servo[TILT_SERVO_GPIO].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    kit.servo[PAN_SERVO_GPIO].angle = 90
    kit.servo[TILT_SERVO_GPIO].angle = 90
    servos_enabled = True
    logger.info("Servos connected and centered using ServoKit.")
except ValueError:
    logger.error("ERROR: Could not initialize ServoKit. Running in 'simulation' mode.")
except Exception as e:
    logger.error(f"Unexpected error during ServoKit initialization: {e}")
    
# --- PEER CONNECTION SETUP ---
# Use STUN servers for network traversal
servers = {
    'iceServers': [
        {'urls': ['stun:stun.l.google.com:19302']},
        # Add your own TURN server for truly robust remote control:
        # {'urls': 'turn:my.turnserver.com:3478', 'username': 'user', 'credential': 'password'} 
    ],
}
pc = RTCPeerConnection(servers)

# --- SERVO MOVEMENT FUNCTION (SAME AS BEFORE) ---
def move_servos(x, y):
    """Maps joystick data (-127 to 127) to servo angles and applies smoothing."""
    global current_pan_angle, current_tilt_angle
    
    if not servos_enabled: return

    try:
        # 1. Calculate Target Angle
        normalized_pan = (x + 127.0) / 254.0
        normalized_tilt = (y + 127.0) / 254.0

        pan_range = PAN_SERVO_MAX_ANGLE - PAN_SERVO_MIN_ANGLE
        tilt_range = TILT_SERVO_MAX_ANGLE - TILT_SERVO_MIN_ANGLE
        
        target_pan_angle = (normalized_pan * pan_range) + PAN_SERVO_MIN_ANGLE
        target_tilt_angle = (normalized_tilt * tilt_range) + TILT_SERVO_MIN_ANGLE
        
        # 2. Apply Motion Smoothing (Lerp)
        new_pan_angle = (current_pan_angle * (1.0 - SMOOTHING_ALPHA)) + (target_pan_angle * SMOOTHING_ALPHA)
        new_tilt_angle = (current_tilt_angle * (1.0 - SMOOTHING_ALPHA)) + (target_tilt_angle * SMOOTHING_ALPHA)
        
        # 3. Clamp values and update state
        final_pan_angle = max(PAN_SERVO_MIN_ANGLE, min(PAN_SERVO_MAX_ANGLE, new_pan_angle))
        final_tilt_angle = max(TILT_SERVO_MIN_ANGLE, min(TILT_SERVO_MAX_ANGLE, new_tilt_angle))

        current_pan_angle = final_pan_angle
        current_tilt_angle = final_tilt_angle

        # Control Servos
        kit.servo[PAN_SERVO_GPIO].angle = final_pan_angle
        kit.servo[TILT_SERVO_GPIO].angle = final_tilt_angle
            
        logger.info(f"Servos -> Pan: {final_pan_angle:.1f}°, Tilt: {final_tilt_angle:.1f}°")

    except Exception as e:
        logger.error(f"Error moving servos via ServoKit: {e}")

# --- WebRTC Event Handlers (MODIFIED ICE CANDIDATE) ---

@pc.on("datachannel")
def on_datachannel(channel):
    logger.info(f"Data channel '{channel.label}' created")
    
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
            
            logger.info(f"Received: J1_X={x}, J1_Y={y}")
            
            # Control the robot: X for Pan, Inverted Y for Tilt
            move_servos(x, -y)
            
        except Exception as e:
            logger.warning(f"Error processing message: {e}")

@pc.on("track")
def on_track(track):
    logger.info(f"Receiving {track.kind} track from web app (Ignoring)")

@pc.on("icecandidate")
def on_icecandidate(candidate):
    """
    MODIFIED: Instead of printing, the candidate is handled within 
    run_firebase_signaling after the Answer is created.
    """
    pass # No need to do anything here, handled later.

@pc.on("connectionstatechange")
async def on_connectionstatechange():
    logger.info(f"Connection state is {pc.connectionState}")
    if pc.connectionState == "failed":
        await pc.close()
        logger.info("Connection failed, shutting down.")
    elif pc.connectionState == "closed":
        logger.info("Connection closed.")


# --- NEW FIREBASE SIGNALING FUNCTION ---
async def run_firebase_signaling():
    """Handles the automatic Firebase signaling as the WebRTC Answerer."""
    if not db:
        logger.error("Cannot run signaling without database connection.")
        return

    room_doc_ref = db.collection('calls').document(ROOM_ID)
    offer_candidates_ref = room_doc_ref.collection('offerCandidates')
    answer_candidates_ref = room_doc_ref.collection('answerCandidates')

    logger.info(f"Waiting for Offer in Firebase Room: {ROOM_ID}...")

    # Event to signal that the Offer has been received
    offer_event = asyncio.Event()
    offer_data = {}
    
    # 1. LISTEN FOR THE OFFER
    # The listener waits for the Web App to create the 'offer' field in the document
    def on_offer_snapshot(doc_snapshot, changes, read_time):
        nonlocal offer_data
        for doc in doc_snapshot:
            data = doc.to_dict()
            if 'offer' in data and not offer_event.is_set():
                offer_data.update(data['offer'])
                offer_event.set()
                logger.info("Offer received from Firebase.")

    offer_watcher = room_doc_ref.on_snapshot(on_offer_snapshot)
    
    await offer_event.wait()
    offer_watcher.unsubscribe() # Stop watching the main doc now

    # 2. PROCESS OFFER AND CREATE ANSWER
    offer = RTCSessionDescription(sdp=offer_data['sdp'], type=offer_data['type'])
    await pc.setRemoteDescription(offer)
    
    # Setup ICE Candidate handler for the Robot (sends to answerCandidates collection)
    pc.onicecandidate = lambda candidate: candidate and answer_candidates_ref.add(candidate.to_dict())

    # Create and set the local Answer description
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    # Write the answer back to the room document
    answer_payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    room_doc_ref.update({'answer': answer_payload})
    logger.info("Answer sent to Firebase. Waiting for connection...")
    
    # 3. LISTEN FOR CLIENT'S ICE CANDIDATES
    # The listener waits for candidates to be posted by the Web App
    def on_offer_candidate_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == 'ADDED':
                candidate_data = change.document.to_dict()
                candidate = RTCIceCandidate(
                    sdpMid=candidate_data.get('sdpMid'),
                    sdpMLineIndex=candidate_data.get('sdpMLineIndex'),
                    candidate=candidate_data.get('candidate')
                )
                # aiortc expects this to be run in the main asyncio loop
                asyncio.run_coroutine_threadsafe(pc.addIceCandidate(candidate), asyncio.get_event_loop())
                logger.info("Added client ICE candidate.")
                
    offer_candidates_ref.on_snapshot(on_offer_candidate_snapshot)
    
    # Wait indefinitely until connection closes or fails
    await asyncio.Event().wait()


async def shutdown(signal, loop):
    """Graceful shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    
    await pc.close()
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    
    logger.info("Cancelling tasks...")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    for sig in (SIGINT, SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, loop)))
        
    try:
        # Run the new firebase signaling function
        loop.run_until_complete(run_firebase_signaling())
    finally:
        logger.info("Cleaning up...")
        if servos_enabled:
            kit.servo[PAN_SERVO_GPIO].angle = 90
            kit.servo[TILT_SERVO_GPIO].angle = 90
            
        loop.close()
        logger.info("Shutdown complete.")