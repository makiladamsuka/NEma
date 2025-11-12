import asyncio
import json
import logging
import sys
import os
from adafruit_servokit import ServoKit
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from signal import SIGINT, SIGTERM

# --- Configuration ---
# PCA9685 Channel Indices (0-15)
PAN_SERVO_GPIO = 0  # Pan (left/right) servo, connected to channel 0
TILT_SERVO_GPIO = 1 # Tilt (up/down) servo, connected to channel 1

# Servo min/max angles (Used for mapping joystick input)
PAN_SERVO_MIN_ANGLE = 50
PAN_SERVO_MAX_ANGLE = 150
TILT_SERVO_MIN_ANGLE = 30
TILT_SERVO_MAX_ANGLE = 150
# --- End Configuration ---


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_pi")

# Global PeerConnection
pc = RTCPeerConnection()
servos_enabled = False

# --- Servo Setup (PCA9685/ServoKit) ---
try:
    # Initialize ServoKit for the PCA9685 (16 channels)
    kit = ServoKit(channels=16) 
    
    # Apply pulse width ranges for better control (if needed for your specific servos)
    kit.servo[PAN_SERVO_GPIO].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    kit.servo[TILT_SERVO_GPIO].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    
    # Center servos on start
    kit.servo[PAN_SERVO_GPIO].angle = 90
    kit.servo[TILT_SERVO_GPIO].angle = 90
    servos_enabled = True
    logger.info("Servos connected and centered using ServoKit.")
    
except ValueError:
    # This exception typically catches I2C initialization failure
    logger.error("ERROR: Could not initialize ServoKit. Check I2C wiring and external power.")
    logger.error("Running in 'simulation' mode. Servos will not move.")
    kit = None
    servos_enabled = False
except Exception as e:
    logger.error(f"An unexpected error occurred during ServoKit initialization: {e}")
    kit = None
    servos_enabled = False


def move_servos(x, y):
    """Maps joystick data (-127 to 127) to servo angles using ServoKit.
    NOTE: The current mapping logic (user-requested) will map (x=0, y=0) to MIN_ANGLE, not the center angle.
    If you want (x=0, y=0) to map to 90 degrees, you should use (x + 127) / 254.0 as the normalized value.
    """
    if not servos_enabled:
        return

    try:
        # --- START USER'S REQUESTED MAPPING LOGIC ---
        # Map X to Pan servo
        # (x / 254) * (max - min) + min
        pan_angle = ((x / 254.0)+1) * (PAN_SERVO_MAX_ANGLE - PAN_SERVO_MIN_ANGLE) + PAN_SERVO_MIN_ANGLE
        
        # Map Y to Tilt servo
        # (y / 254) * (max - min) + min
        tilt_angle = ((y / 254.0)+1) * (TILT_SERVO_MAX_ANGLE - TILT_SERVO_MIN_ANGLE) + TILT_SERVO_MIN_ANGLE
        # --- END USER'S REQUESTED MAPPING LOGIC ---
        
        # Clamp values to be safe
        pan_angle = max(PAN_SERVO_MIN_ANGLE, min(PAN_SERVO_MAX_ANGLE, pan_angle))
        tilt_angle = max(TILT_SERVO_MIN_ANGLE, min(TILT_SERVO_MAX_ANGLE, tilt_angle))

        # Control Servos using ServoKit
        kit.servo[PAN_SERVO_GPIO].angle = pan_angle
        kit.servo[TILT_SERVO_GPIO].angle = tilt_angle
            
        logger.info(f"Servos -> Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")

    except Exception as e:
        logger.error(f"Error moving servos via ServoKit: {e}")


# --- WebRTC Event Handlers ---

@pc.on("datachannel")
def on_datachannel(channel):
    logger.info(f"Data channel '{channel.label}' created")
    
    # We must check the label
    if channel.label != "controls":
        logger.warning(f"Ignoring data channel: {channel.label}")
        return

    @channel.on("message")
    def on_message(message):
        try:
            data = json.loads(message)
            
            # Get the 'j1' object (joystick 1) for pan/tilt
            j1_data = data.get('j1', {})
            x = j1_data.get('x', 0)
            y = j1_data.get('y', 0)
            
            # You can also get other data if needed
            # j2_data = data.get('j2', {})
            switches = data.get('sw', 0)

            logger.info(f"Received: J1_X={x}, J1_Y={y}, SW={switches}")
            
            # This is where we control the robot!
            move_servos(x, y)
            
        except Exception as e:
            logger.warning(f"Error processing message: {e}")
            logger.warning(f"Raw message: {message}")

@pc.on("track")
def on_track(track):
    logger.info(f"Receiving {track.kind} track from web app (Ignoring)")

@pc.on("icecandidate")
def on_icecandidate(candidate):
    if candidate:
        logger.info("--- NEW ROBOT ICE CANDIDATE (Copy to Web App) ---")
        print(json.dumps(candidate.to_dict()))

@pc.on("connectionstatechange")
async def on_connectionstatechange():
    logger.info(f"Connection state is {pc.connectionState}")
    if pc.connectionState == "failed":
        await pc.close()
        logger.info("Connection failed, shutting down.")
    elif pc.connectionState == "closed":
        logger.info("Connection closed.")


async def run_signaling():
    """Handles the manual copy-paste signaling."""
    loop = asyncio.get_running_loop()
    
    logger.info("No camera or microphone tracks are being streamed.")

    
    # 2. Get Offer from Web App
    logger.info("Paste Offer from Web App (single JSON object):")
    offer_json = await loop.run_in_executor(None, sys.stdin.readline)
    offer = json.loads(offer_json)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer['sdp'], type=offer['type']))
    logger.info("Offer received.")
    
    # 3. Get Web App ICE Candidates
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

    # 4. Create and send Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    logger.info("--- YOUR ANSWER (Copy to Web App) ---")
    print(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
    
    logger.info("Waiting for connection... (Copy any new Robot ICE candidates to the Web App)")
    # Wait indefinitely until connection closes
    await asyncio.Event().wait()


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
    loop.stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # Add signal handlers for graceful shutdown
    for sig in (SIGINT, SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, loop)))
        
    try:
        loop.run_until_complete(run_signaling())
    finally:
        logger.info("Cleaning up...")
        # Center servos on exit if enabled
        if servos_enabled:
            # Setting angle to 90 to ensure servos are in a neutral position
            kit.servo[PAN_SERVO_GPIO].angle = 90
            kit.servo[TILT_SERVO_GPIO].angle = 90
            
        loop.close()
        logger.info("Shutdown complete.")