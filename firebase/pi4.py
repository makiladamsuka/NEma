#added servo control via PCA9685 

import asyncio
import json
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import os

# --- Hardware Imports ---
# Import the necessary libraries for PCA9685 control
# NOTE: This requires the libraries installed in Step 1.
try:
    import board
    import adafruit_pca9685
    from adafruit_motor import servo
except ImportError:
    print("WARNING: Hardware control libraries not found.")
    print("Install with: pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-busdevice")
    
# --- PCA9685 Globals ---
PCA = None
PAN_SERVO = None
TILT_SERVO = None

# --- Configuration & Globals ---
# CRITICAL: REPLACE with the correct path to your Firebase service account key
FIREBASE_KEY_PATH = os.path.expanduser('/home/nema/Documents/NEma/firebase/serviceAccountKey.json') 
ROOM_ID = 'esp32_robot_room' 
CALLS_COLLECTION = 'robot_calls'
CONNECTION_ESTABLISHED = False

# Global variables for async loop and PeerConnection
MAIN_EVENT_LOOP = None
pc = None 

# --- Hardware Initialization ---

def initialize_pca9685():
    """Initializes the I2C connection to the PCA9685 and sets up the servos."""
    global PCA, PAN_SERVO, TILT_SERVO
    try:
        # 1. Initialize I2C Bus
        i2c = board.I2C()
        
        # 2. Initialize PCA9685 (default address is 0x40)
        PCA = adafruit_pca9685.PCA9685(i2c)
        PCA.frequency = 50  # Standard frequency for hobby servos

        # 3. Define Servos on specific channels (adjust these pins for your robot)
        PAN_CHANNEL = 0
        TILT_CHANNEL = 1
        
        # Min and Max pulse width in microseconds (usually 500 to 2500)
        # This defines the 0 to 180 degree limits for the servo pulse.
        SERVO_MIN_PULSE = 500
        SERVO_MAX_PULSE = 2500

        # Initialize the Servo objects
        PAN_SERVO = servo.Servo(
            PCA.channels[PAN_CHANNEL], 
            min_pulse=SERVO_MIN_PULSE, 
            max_pulse=SERVO_MAX_PULSE
        )
        TILT_SERVO = servo.Servo(
            PCA.channels[TILT_CHANNEL], 
            min_pulse=SERVO_MIN_PULSE, 
            max_pulse=SERVO_MAX_PULSE
        )
        
        # Set servos to the center position (90 degrees)
        PAN_SERVO.angle = 90
        TILT_SERVO.angle = 90
        
        print("PCA9685 and Servos Initialized successfully.")
        return True
    
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize PCA9685: {e}")
        return False


# --- Firebase Setup ---
try:
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(f"Firebase key not found at {FIREBASE_KEY_PATH}")
        
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    exit()
except Exception as e:
    print(f"FATAL ERROR initializing Firebase: {e}")
    exit()

db = firestore.client()
call_doc_ref = db.collection(CALLS_COLLECTION).document(ROOM_ID)

# --- Robot Safety and Control ---

def stop_robot_safety_system():
    """
    SAFETY FUNCTION: Immediately stops all motors/actuators and centers servos.
    This function MUST be synchronous (no 'await').
    """
    global PCA, PAN_SERVO, TILT_SERVO
    
    print("\n\n***************************************************")
    print("*** SAFETY STOP ACTIVATED: Browser Disconnected ***")
    print("***************************************************")
    
    # Reset servos to a safe, neutral, or parking position (90 degrees)
    if PAN_SERVO and TILT_SERVO:
        PAN_SERVO.angle = 90
        TILT_SERVO.angle = 90
        print("Servos centered.")
    
    # ----------------------------------------------------
    # TODO: Add motor stop logic for other drive motors here!
    # ----------------------------------------------------

def handle_joystick_data(message):
    """
    Handles incoming JSON data from the WebRTC DataChannel and controls servos.
    """
    global PAN_SERVO, TILT_SERVO
    
    try:
        data = json.loads(message)
        j1 = data.get('j1', {})
        j2 = data.get('j2', {})
        sw = data.get('sw', 0)
        
        j1_x = j1.get('x', 0)
        j1_y = j1.get('y', 0)
        # j2_x and j2_y can be used for other axes or linear motors
        
        # 1. MAP JOYSTICK VALUE (-127 to 127) to SERVO ANGLE (0 to 180)
        
        # Pan Angle: Map J1_X (-127=Left, 127=Right) to (0° to 180°)
        # We normalize the joystick value (0 to 1) and then map to the 0-180 range.
        # Ensure the value is within the -127 to 127 limits
        pan_norm = max(-127, min(127, j1_x))
        # Formula: (Joystick + 127) / 254 * 180 
        pan_angle = int((pan_norm + 127) / 254 * 180)
        
        # Tilt Angle: Map J1_Y (-127=Down, 127=Up) to (0° to 180°)
        tilt_norm = max(-127, min(127, j1_y))
        # Depending on your servo mounting, you might need to invert this.
        # tilt_angle = 180 - int((tilt_norm + 127) / 254 * 180) # Example of inverted mapping
        tilt_angle = int((tilt_norm + 127) / 254 * 180)
        
        # 2. APPLY LIMITS (Optional, depending on your pan/tilt mechanism)
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))


        # 3. CONTROL SERVOS (CRITICAL STEP)
        if PAN_SERVO and TILT_SERVO:
            PAN_SERVO.angle = pan_angle
            TILT_SERVO.angle = tilt_angle
        
        
        print("\n--- Received Control Data ---")
        print(f"J1 (Pan/Tilt): X={j1_x}, Y={j1_y}")
        print(f"Switches (Bitmask): {sw}")
        print(f"--> SERVO OUTPUT: Pan={pan_angle}° (Ch 0), Tilt={tilt_angle}° (Ch 1)")

    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")
    except Exception as e:
        print(f"Error processing data: {e}")


# --- WebRTC & Signaling Core ---

# Flag to ensure the offer is processed only once per new connection attempt
OFFER_PROCESSED = False

async def cleanup_firebase_room():
    """
    Deletes the signaling room in Firestore after disconnection.
    Uses asyncio.to_thread to safely handle the synchronous Firebase Admin SDK.
    """
    print("Cleaning up Firebase room...")
    try:
        # FIX: Use asyncio.to_thread to wrap the synchronous call
        await asyncio.to_thread(call_doc_ref.delete)
        
        print("Room document deleted. Ready for next browser connection.")
    except Exception as e:
        print(f"Error during Firebase cleanup: {e}")


def handle_snapshot_in_thread(col_snapshot):
    """
    Callback executed in the synchronous Firebase thread. 
    Safely schedules the async processing function onto the main event loop.
    """
    global MAIN_EVENT_LOOP
    
    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(
            process_offer_snapshot(col_snapshot),
            MAIN_EVENT_LOOP
        )

def handle_ice_snapshot_in_thread(pc_instance, col_snapshot):
    """
    Callback for ICE candidates, safely scheduled to the main event loop.
    """
    global MAIN_EVENT_LOOP
    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(
            add_ice_candidates(pc_instance, col_snapshot),
            MAIN_EVENT_LOOP
        )

async def add_ice_candidates(pc_instance, col_snapshot):
    for change in col_snapshot:
        if change.type.name == 'ADDED':
            candidate_data = change.document.to_dict()
            try:
                if candidate_data.get('candidate'):
                    candidate = RTCIceCandidate(**candidate_data)
                    await pc_instance.addIceCandidate(candidate)
            except Exception as e:
                print(f"Error adding ICE candidate: {e}")


async def process_offer_snapshot(col_snapshot):
    global OFFER_PROCESSED, CONNECTION_ESTABLISHED, pc
    
    # Block processing if already connected 
    if CONNECTION_ESTABLISHED:
        return

    # Check for offer data
    try:
        doc = col_snapshot[0]
        data = doc.to_dict()
    except IndexError:
        return 

    if 'offer' in data and not OFFER_PROCESSED:
        OFFER_PROCESSED = True # Mark as processing to avoid race conditions
        print("Offer received! Processing...")
        
        offer = RTCSessionDescription(
            sdp=data['offer']['sdp'],
            type=data['offer']['type'])

        # Create a new PeerConnection for every new call
        pc = RTCPeerConnection() 
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"Data channel opened: {channel.label}")
            
            @channel.on("message")
            def on_message(message):
                handle_joystick_data(message)

            @channel.on("open")
            def on_open():
                global CONNECTION_ESTABLISHED
                CONNECTION_ESTABLISHED = True
                print("WebRTC DataChannel OPENED. Starting control loop.")

            @channel.on("close")
            def on_close():
                global CONNECTION_ESTABLISHED, OFFER_PROCESSED
                print("WebRTC DataChannel CLOSED.")
                
                # 1. CRITICAL: STOP THE ROBOT IMMEDIATELY
                stop_robot_safety_system() 
                
                # 2. Reset connection flags to allow a new incoming call
                CONNECTION_ESTABLISHED = False
                OFFER_PROCESSED = False
                
                # 3. Schedule ASYNC cleanup (Thread-safe)
                if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(
                        cleanup_firebase_room(),
                        MAIN_EVENT_LOOP
                    )


        # --- ICE Candidate Handling ---
        offer_candidates_ref = call_doc_ref.collection('offerCandidates')
        offer_candidates_ref.on_snapshot(lambda col_snapshot, changes, read_time: 
            handle_ice_snapshot_in_thread(pc, col_snapshot))

        answer_candidates_ref = call_doc_ref.collection('answerCandidates')
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                answer_candidates_ref.add(candidate.toJSON())

        # 3. Create and Send Answer
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        answer_data = {
            'answer': {
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }
        }
        await call_doc_ref.update(answer_data)
        print("Answer sent to Firestore. Connection process initiated.")


async def run_answerer():
    global MAIN_EVENT_LOOP
    print("WebRTC Robot Answerer starting in CONTINUOUS LISTEN mode...")
    
    # --- HARDWARE INIT ---
    if not initialize_pca9685():
        print("ERROR: PCA9685 initialization failed. Running in debug mode (no servo control).")
    
    # Get the current running event loop in the main thread
    MAIN_EVENT_LOOP = asyncio.get_running_loop() 

    # Start watching for changes in the room document (new Offer)
    call_doc_ref.on_snapshot(lambda col_snapshot, changes, read_time: 
                             handle_snapshot_in_thread(col_snapshot))

    print("Listening for new WebRTC Offers...")
    
    # The script now runs indefinitely, waiting for new calls.
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(run_answerer())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down gracefully...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")