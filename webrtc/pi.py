import asyncio
import json
import logging
import sys
import os

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer
# Use gpiozero for servos. Make sure 'pigpiod' is running!
# (sudo systemctl start pigpiod)
from gpiozero import AngularServo
from signal import SIGINT, SIGTERM

# --- Configuration ---
# Adjust GPIO pins as needed
PAN_SERVO_GPIO = 0  # Pan (left/right) servo
TILT_SERVO_GPIO = 1 # Tilt (up/down) servo

#pan = 50 150 tilt 30 150

# Servo min/max angles
PAN_SERVO_MIN_ANGLE = 50
PAN_SERVO_MAX_ANGLE = 150
TILT_SERVO_MIN_ANGLE = 30
TILT_SERVO_MAX_ANGLE = 150

# Camera and Microphone (Linux device names)
# Use 'ls /dev/video*' to find your camera
PI_CAMERA = "/dev/video0" 
# Use 'arecord -l' to find your microphone's card/device (e.g., "hw:1,0")
PI_MICROPHONE = "hw:1,0" 

# --- End Configuration ---


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_pi")

# Global PeerConnection
pc = RTCPeerConnection()

# --- Servo Setup ---
try:
    pan_servo = AngularServo(
        PAN_SERVO_GPIO, 
        min_angle=PAN_SERVO_MIN_ANGLE, 
        max_angle=PAN_SERVO_MAX_ANGLE
    )
    tilt_servo = AngularServo(
        TILT_SERVO_GPIO, 
        min_angle=TILT_SERVO_MIN_ANGLE, 
        max_angle=TILT_SERVO_MAX_ANGLE
    )
    # Center servos on start
    pan_servo.angle = 90
    tilt_servo.angle = 90
    logger.info("Servos connected and centered.")
except Exception as e:
    logger.error(f"Failed to initialize servos. Is 'pigpiod' running?")
    logger.error(f"Error: {e}")
    logger.error("Running in 'simulation' mode. Servos will not move.")
    pan_servo = None
    tilt_servo = None


def move_servos(x, y):
    """Maps joystick data (-127 to 127) to servo angles."""
    try:
        # Map X to Pan servo
        # (x / 127.0) gives a ratio from -1.0 to 1.0
        pan_angle = (x / 254) * (PAN_SERVO_MAX_ANGLE - PAN_SERVO_MIN_ANGLE) + PAN_SERVO_MIN_ANGLE
        
        # Map Y to Tilt servo
        # (y / 127.0) gives a ratio from -1.0 to 1.0
        # Invert Y so "up" on joystick is "up" on head
        tilt_angle = (y / 254) * (TILT_SERVO_MAX_ANGLE - TILT_SERVO_MIN_ANGLE) + TILT_SERVO_MIN_ANGLE
        
        # Clamp values to be safe
        pan_angle = max(PAN_SERVO_MIN_ANGLE, min(PAN_SERVO_MAX_ANGLE, pan_angle))
        tilt_angle = max(TILT_SERVO_MIN_ANGLE, min(TILT_SERVO_MAX_ANGLE, tilt_angle))

        if pan_servo:
            pan_servo.angle = pan_angle
        if tilt_servo:
            tilt_servo.angle = tilt_angle
            
        logger.info(f"Servos -> Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")

    except Exception as e:
        logger.error(f"Error moving servos: {e}")


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
            
            # --- THIS IS THE UPDATED LOGIC ---
            # Get the 'j1' object (joystick 1) for pan/tilt
            j1_data = data.get('j1', {})
            x = j1_data.get('x', 0)
            y = j1_data.get('y', 0)
            
            # You can also get other data if needed
            j2_data = data.get('j2', {})
            switches = data.get('sw', 0)

            logger.info(f"Received: J1_X={x}, J1_Y={y}, SW={switches}")
            
            # This is where we control the robot!
            move_servos(x, y)
            
            # Example: Use switch 1 (bit 0) for something
            # if (switches & (1 << 0)):
            #    logger.info("Switch 1 is PRESSED!")
            
        except Exception as e:
            logger.warning(f"Error processing message: {e}")
            logger.warning(f"Raw message: {message}")

@pc.on("track")
def on_track(track):
    logger.info(f"Receiving {track.kind} track from web app")
    # Here you could, for example, play the audio track
    # or display the video track on the Pi's display.
    if track.kind == "audio":
        pass
    if track.kind == "video":
        pass

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
    
    # 1. Add Pi's Camera and Mic tracks
    try:
        if os.path.exists(PI_CAMERA):
            camera = MediaPlayer(PI_CAMERA, format="v4l2", options={'video_size': '640x480'})
            pc.addTrack(camera.video)
            logger.info(f"Streaming from camera: {PI_CAMERA}")
        else:
            logger.warning(f"Camera device not found: {PI_CAMERA}")
            
        if PI_MICROPHONE:
            microphone = MediaPlayer(PI_MICROPHONE, format="alsa")
            pc.addTrack(microphone.audio)
            logger.info(f"Streaming from microphone: {PI_MICROPHONE}")
    except Exception as e:
        logger.error(f"Error opening media devices: {e}")
        logger.error("Check camera/mic paths and 'arecord -l' for ALSA devices.")

    
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
        if pan_servo:
            pan_servo.angle = 0 # Center servos on exit
            pan_servo.close()
        if tilt_servo:
            tilt_servo.angle = 0
            tilt_servo.close()
        loop.close()
        logger.info("Shutdown complete.")