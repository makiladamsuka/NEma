import asyncio
import json
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import os

# --- Configuration & Globals ---
# REPLACE with the correct path to your Firebase service account key
FIREBASE_KEY_PATH = os.path.expanduser('/home/nema/Documents/NEma/firebase/serviceAccountKey.json') 
ROOM_ID = 'esp32_robot_room' 
CALLS_COLLECTION = 'robot_calls'
CONNECTION_ESTABLISHED = False

# Global variables for async loop and PeerConnection
MAIN_EVENT_LOOP = None
pc = None 

# --- Firebase Setup ---
try:
    # Use os.path.expanduser for robust path handling
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
    SAFETY FUNCTION: Immediately stops all motors/actuators.
    This function MUST be synchronous (no 'await').
    """
    print("\n\n***************************************************")
    print("*** SAFETY STOP ACTIVATED: Browser Disconnected ***")
    print("***************************************************")
    
    # ----------------------------------------------------
    # TODO: IMPLEMENT YOUR SPECIFIC HARDWARE STOP COMMANDS HERE!
    # Example concepts (replace with actual GPIO/PWM code for your Pi):
    
    # 1. Stop all driving motors immediately
    # motor_controller.set_speed(0) 
    
    # 2. Return servos to a safe, neutral, or parking position (e.g., 90 degrees)
    # servo_controller.set_angle('pan', 90) 
    # ----------------------------------------------------

def handle_joystick_data(message):
    """
    Handles incoming JSON data from the WebRTC DataChannel.
    """
    try:
        data = json.loads(message)
        j1 = data.get('j1', {})
        j2 = data.get('j2', {})
        sw = data.get('sw', 0)
        
        j1_x = j1.get('x', 0)
        j1_y = j1.get('y', 0)
        j2_x = j2.get('x', 0)
        j2_y = j2.get('y', 0)

        print("\n--- Received Control Data ---")
        print(f"J1 (Pan/Tilt): X={j1_x}, Y={j1_y}")
        print(f"J2 (Auxiliary): X={j2_x}, Y={j2_y}")
        print(f"Switches (Bitmask): {sw} (SW1: {(sw & 1) > 0}, SW2: {(sw & 2) > 0})")
        
        # Example: Map -127 to 127 to an angle range like -90 to +90
        pan_angle = int(j1_x / 127 * 90)  
        tilt_angle = int(j1_y / 127 * 90) 
        print(f"Calculated Angles: Pan={pan_angle}° (from J1 X), Tilt={tilt_angle}° (from J1 Y)")

    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")
    except Exception as e:
        print(f"Error processing data: {e}")

# --- WebRTC & Signaling Core ---

# Flag to ensure the offer is processed only once
OFFER_PROCESSED = False

async def cleanup_firebase_room():
    """Deletes the signaling room in Firestore after disconnection."""
    print("Cleaning up Firebase room...")
    try:
        await call_doc_ref.delete()
        print("Room document deleted. Ready for next session.")
    except Exception as e:
        print(f"Error during Firebase cleanup: {e}")


def handle_snapshot_in_thread(col_snapshot):
    """
    Callback executed in the synchronous Firebase thread. 
    Safely schedules the async processing function onto the main event loop.
    """
    global MAIN_EVENT_LOOP
    
    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
        # Use run_coroutine_threadsafe to safely schedule the async function
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
                # Add candidate if it exists
                if candidate_data.get('candidate'):
                    candidate = RTCIceCandidate(**candidate_data)
                    await pc_instance.addIceCandidate(candidate)
            except Exception as e:
                # Often just redundant data, but worth noting
                print(f"Error adding ICE candidate: {e}")


async def process_offer_snapshot(col_snapshot):
    global OFFER_PROCESSED, CONNECTION_ESTABLISHED, pc
    
    if OFFER_PROCESSED or CONNECTION_ESTABLISHED:
        return

    # Check for offer data
    try:
        doc = col_snapshot[0]
        data = doc.to_dict()
    except IndexError:
        return 

    if 'offer' in data and not CONNECTION_ESTABLISHED:
        OFFER_PROCESSED = True
        print("Offer received! Processing...")
        
        offer = RTCSessionDescription(
            sdp=data['offer']['sdp'],
            type=data['offer']['type'])

        pc = RTCPeerConnection()
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            global CONNECTION_ESTABLISHED
            print(f"Data channel opened: {channel.label}")
            
            @channel.on("message")
            def on_message(message):
                handle_joystick_data(message)

            @channel.on("open")
            def on_open():
                CONNECTION_ESTABLISHED = True
                print("WebRTC DataChannel OPENED. Starting control loop.")
                # Reset flag for potential future re-connections without script restart
                global OFFER_PROCESSED
                OFFER_PROCESSED = False 

            @channel.on("close")
            def on_close():
                CONNECTION_ESTABLISHED = False
                print("WebRTC DataChannel CLOSED.")
                
                # 1. CRITICAL: STOP THE ROBOT IMMEDIATELY
                stop_robot_safety_system() 
                
                # 2. Schedule ASYNC cleanup and STOP the loop
                if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
                    # Schedule cleanup to run on the main loop
                    asyncio.run_coroutine_threadsafe(
                        cleanup_firebase_room(),
                        MAIN_EVENT_LOOP
                    )
                    
                    # 3. Stop the main asyncio loop to exit the script
                    MAIN_EVENT_LOOP.stop() 


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
    global CONNECTION_ESTABLISHED, MAIN_EVENT_LOOP
    print("WebRTC Robot Answerer starting...")
    
    # Get the current running event loop in the main thread
    MAIN_EVENT_LOOP = asyncio.get_running_loop() 

    # Start watching for changes in the room document (new Offer)
    call_doc_ref.on_snapshot(lambda col_snapshot, changes, read_time: 
                             handle_snapshot_in_thread(col_snapshot))

    print("Waiting for WebRTC Offer in Firestore...")
    
    # Keep the script running until CONNECTION_ESTABLISHED becomes False 
    # (which happens when the connection closes and the loop is stopped)
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        # aiortc expects asyncio.run to start the event loop
        asyncio.run(run_answerer())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
    except RuntimeError as e:
        # This occurs normally when MAIN_EVENT_LOOP.stop() is called
        if "stopped" in str(e):
            print("Cleanup complete. Program exited gracefully.")
        else:
            print(f"An unexpected RuntimeError occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")