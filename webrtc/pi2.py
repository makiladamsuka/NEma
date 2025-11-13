# pi.py
# This is the complete, modified file for the Raspberry Pi.

import asyncio
import json
import logging
import sys
import os
import websockets  # Added for automated signaling

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

# --- Signaling Server URL ---
# This should point to your signaling server (which is also on the Pi)
SIGNALING_SERVER_URL = "ws://localhost:8765/pi"

# --- End Configuration ---


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_pi")

# Global PeerConnection
pc = RTCPeerConnection()

# Global WebSocket connection
ws = None

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
        # 1. Normalize the joystick value from [-127, 127] to [0.0, 1.0]
        # (x + 127.0) gives a range of [0, 254]
        # / 254.0 gives a range of [0.0, 1.0]
        pan_normalized = (x + 127.0) / 254.0
        tilt_normalized = (y + 127.0) / 254.0 # 0.0 is "down", 1.0 is "up"

        # 2. Map the [0.0, 1.0] range to the servo's angle range
        pan_angle = pan_normalized * (PAN_SERVO_MAX_ANGLE - PAN_SERVO_MIN_ANGLE) + PAN_SERVO_MIN_ANGLE
        
        # We assume MIN_ANGLE (30) is "head up" and MAX_ANGLE (150) is "head down".
        # This means we must INVERT the tilt mapping.
        # When joystick is "up" (y=127, norm=1.0), we want MIN_ANGLE (30).
        # (1.0 - 1.0) * range + 30 = 30
        # When joystick is "down" (y=-127, norm=0.0), we want MAX_ANGLE (150).
        # (1.0 - 0.0) * range + 30 = 150
        tilt_angle = (1.0 - tilt_normalized) * (TILT_SERVO_MAX_ANGLE - TILT_SERVO_MIN_ANGLE) + TILT_SERVO_MIN_ANGLE
        
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
    # This function is NOT async, so we can't 'await'
    # We pass the candidate to the main async loop to send
    if candidate:
        logger.info("Generated new ICE candidate")
        asyncio.create_task(send_to_server({
            'type': 'ice-candidate',
            'candidate': candidate.to_dict()
        }))

@pc.on("connectionstatechange")
async def on_connectionstatechange():
    logger.info(f"Connection state is {pc.connectionState}")
    if pc.connectionState == "failed":
        await pc.close()
        logger.info("Connection failed, shutting down.")
    elif pc.connectionState == "closed":
        logger.info("Connection closed.")


# --- Signaling Functions ---

async def send_to_server(message_dict):
    """Helper function to send JSON messages to the server."""
    if ws:
        await ws.send(json.dumps(message_dict))

async def run_webrtc():
    """Handles the automatic signaling."""
    global ws
    
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

    
    # 2. Connect to the signaling server
    logger.info(f"Connecting to signaling server at {SIGNALING_SERVER_URL}")
    try:
        async with websockets.connect(SIGNALING_SERVER_URL) as websocket:
            ws = websocket # Store websocket globally for on_icecandidate
            logger.info("Connected to signaling server.")
            
            # 3. Listen for messages from the server
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'offer':
                    logger.info("Received Offer")
                    offer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
                    await pc.setRemoteDescription(offer)
                    
                    # Create and send answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    logger.info("Sending Answer")
                    await send_to_server({
                        'type': answer.type,
                        'sdp': answer.sdp
                    })
                    
                elif data['type'] == 'ice-candidate':
                    logger.info("Received ICE candidate")
                    candidate_json = data.get('candidate', {})
                    candidate = RTCIceCandidate(
                        sdpMid=candidate_json.get('sdpMid'),
                        sdpMLineIndex=candidate_json.get('sdpMLineIndex'),
                        candidate=candidate_json.get('candidate')
                    )
                    await pc.addIceCandidate(candidate)
                
                elif data['type'] == 'answer':
                     # This shouldn't happen for the Pi, but good to log
                     logger.warning("Received an 'answer', which was unexpected.")
                     
    except Exception as e:
        logger.error(f"Connection to signaling server failed: {e}")
        logger.error("Is the signaling_server.py running?")


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
        loop.run_until_complete(run_webrtc())
    finally:
        logger.info("Cleaning up...")
        if pan_servo:
            pan_servo.angle = 90 # Center servos on exit
            pan_servo.close()
        if tilt_servo:
            tilt_servo.angle = 90
            tilt_servo.close()
        loop.close()
        logger.info("Shutdown complete.")