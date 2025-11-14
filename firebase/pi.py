import asyncio
import json
import firebase_admin
from firebase_admin import credentials, firestore
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

# --- Configuration ---
# Replace with the path to your Firebase service account key
FIREBASE_KEY_PATH = '/home/nema/Documents/NEma/firebase/serviceAccountKey.json' 
ROOM_ID = 'esp32_robot_room' 
CALLS_COLLECTION = 'robot_calls'
CONNECTION_ESTABLISHED = False

# --- Firebase Setup ---
# Use a service account
try:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
except FileNotFoundError:
    print(f"ERROR: Firebase key not found at {FIREBASE_KEY_PATH}. Please check the path.")
    exit()

db = firestore.client()
call_doc_ref = db.collection(CALLS_COLLECTION).document(ROOM_ID)


def handle_joystick_data(message):
    """
    Handles incoming JSON data from the WebRTC DataChannel.
    """
    try:
        data = json.loads(message)
        j1 = data.get('j1', {})
        j2 = data.get('j2', {})
        sw = data.get('sw', 0)
        
        # J1 (Pan/Tilt)
        j1_x = j1.get('x', 0)
        j1_y = j1.get('y', 0)
        
        # J2 (Auxiliary)
        j2_x = j2.get('x', 0)
        j2_y = j2.get('y', 0)

        # --- Your Pi Control Logic Goes Here ---
        
        # ANALOGY: Think of the joystick values (j1_x, j1_y) like the 
        # **directions a remote control car receives.** The Python script 
        # is the **receiver/engine** in the car that takes these directions 
        # and translates them into actual movement commands (like PWM signals for motors).
        
        # The values range from -127 to 127.
        
        print("\n--- Received Control Data ---")
        print(f"J1 (Pan/Tilt): X={j1_x}, Y={j1_y}")
        print(f"J2 (Auxiliary): X={j2_x}, Y={j2_y}")
        print(f"Switches (Bitmask): {sw} (SW1: {(sw & 1) > 0}, SW2: {(sw & 2) > 0})")
        
        # Example of calculating an angle/duty cycle (simple motor control)
        # Note: In a real robot, you would map these to PWM duty cycles or servo angles.
        pan_angle = int(j1_x / 127 * 90)  # Map -127 to 127 to an angle range like -90 to +90
        tilt_angle = int(j1_y / 127 * 90) # Map -127 to 127 to an angle range like -90 to +90
        print(f"Calculated Angles: Pan={pan_angle}° (from J1 X), Tilt={tilt_angle}° (from J1 Y)")

    except json.JSONDecodeError as e:
        print(f"Received non-JSON message: {message}")
    except Exception as e:
        print(f"Error processing data: {e}")

async def run_answerer():
    global CONNECTION_ESTABLISHED
    print("Waiting for WebRTC Offer in Firestore...")

    # --- Listen for Offer ---
    # Get initial Offer data (blocking call, runs once)
    offer_candidates_ref = call_doc_ref.collection('offerCandidates')
    
    # Watch for changes in the document (new Offer)
    offer_watch = call_doc_ref.on_snapshot(lambda col_snapshot, changes, read_time: asyncio.ensure_future(process_offer_snapshot(col_snapshot)))

    # Keep the script running
    while not CONNECTION_ESTABLISHED:
        await asyncio.sleep(1)

    # After connection, keep running to receive data
    while CONNECTION_ESTABLISHED:
        await asyncio.sleep(1)
        
    print("Connection closed. Exiting.")

# Use a global flag to ensure the process_offer_snapshot runs only once per offer
# In a real app, you might use a lock or a more complex state machine
OFFER_PROCESSED = False

async def process_offer_snapshot(col_snapshot):
    global OFFER_PROCESSED, CONNECTION_ESTABLISHED
    if OFFER_PROCESSED or CONNECTION_ESTABLISHED:
        return

    doc = col_snapshot[0]
    data = doc.to_dict()
    
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

            @channel.on("close")
            def on_close():
                CONNECTION_ESTABLISHED = False
                print("WebRTC DataChannel CLOSED.")

        # --- ICE Candidate Gathering (Answerer) ---
        # 1. Listen for Caller's ICE Candidates
        offer_candidates_ref = call_doc_ref.collection('offerCandidates')
        offer_candidates_ref.on_snapshot(lambda col_snapshot, changes, read_time: 
            asyncio.ensure_future(add_ice_candidates(pc, col_snapshot)))

        # 2. Collect own ICE Candidates and write them to Firestore
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


async def add_ice_candidates(pc, col_snapshot):
    for change in col_snapshot:
        if change.type.name == 'ADDED':
            candidate_data = change.document.to_dict()
            try:
                candidate = RTCIceCandidate(**candidate_data)
                await pc.addIceCandidate(candidate)
            except Exception as e:
                print(f"Error adding ICE candidate: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_answerer())
    except KeyboardInterrupt:
        print("Program interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")