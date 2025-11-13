import asyncio
import json
import logging
import sys
import os
import websockets
from adafruit_servokit import ServoKit
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from signal import SIGINT, SIGTERM

# --- CONFIGURATION ---
PAN_SERVO_GPIO = 0  # Pan (left/right) servo, connected to channel 0
TILT_SERVO_GPIO = 1 # Tilt (up/down) servo, connected to channel 1

PAN_SERVO_MIN_ANGLE = 50
PAN_SERVO_MAX_ANGLE = 150
TILT_SERVO_MIN_ANGLE = 30
TILT_SERVO_MAX_ANGLE = 150

# WebSocket Server Configuration (Run this on the Pi's IP)
SIGNALING_SERVER_ADDRESS = "0.0.0.0" 
SIGNALING_SERVER_PORT = 8765 

# Smoothing Factor (Alpha): 0.15 for smooth but responsive movement
SMOOTHING_ALPHA = 0.15 
# ---------------------


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot_pi")


# --- GLOBAL STATE ---
pc = RTCPeerConnection()
servos_enabled = False
current_pan_angle = 90.0
current_tilt_angle = 90.0
# --------------------


try:
    # Initialize ServoKit for the PCA9685 (16 channels)
    kit = ServoKit(channels=16) 
    
    # Apply pulse width ranges for better control 
    kit.servo[PAN_SERVO_GPIO].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    kit.servo[TILT_SERVO_GPIO].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    
    # Center servos on start
    kit.servo[PAN_SERVO_GPIO].angle = 90
    kit.servo[TILT_SERVO_GPIO].angle = 90
    
    servos_enabled = True
    logger.info("Servos connected and centered using ServoKit.")
    
except ValueError:
    logger.error("ERROR: Could not initialize ServoKit. Check I2C wiring and external power.")
    logger.error("Running in 'simulation' mode. Servos will not move.")
    kit = None
    servos_enabled = False
except Exception as e:
    logger.error(f"An unexpected error occurred during ServoKit initialization: {e}")
    kit = None
    servos_enabled = False


def move_servos(x, y):
    """Maps joystick data (-127 to 127) to servo angles and applies smoothing."""
    global current_pan_angle, current_tilt_angle 
    
    if not servos_enabled:
        return

    try:
        # 1. CALCULATE TARGET ANGLE (0 maps to 90 degrees)
        normalized_pan = (x + 127.0) / 254.0
        normalized_tilt = (y + 127.0) / 254.0

        pan_range = PAN_SERVO_MAX_ANGLE - PAN_SERVO_MIN_ANGLE
        tilt_range = TILT_SERVO_MAX_ANGLE - TILT_SERVO_MIN_ANGLE
        
        target_pan_angle = (normalized_pan * pan_range) + PAN_SERVO_MIN_ANGLE
        target_tilt_angle = (normalized_tilt * tilt_range) + TILT_SERVO_MIN_ANGLE
        
        # 2. APPLY MOTION SMOOTHING (Lerp)
        new_pan_angle = (current_pan_angle * (1.0 - SMOOTHING_ALPHA)) + (target_pan_angle * SMOOTHING_ALPHA)
        new_tilt_angle = (current_tilt_angle * (1.0 - SMOOTHING_ALPHA)) + (target_tilt_angle * SMOOTHING_ALPHA)
        
        
        # 3. Clamp values and update state
        final_pan_angle = max(PAN_SERVO_MIN_ANGLE, min(PAN_SERVO_MAX_ANGLE, new_pan_angle))
        final_tilt_angle = max(TILT_SERVO_MIN_ANGLE, min(TILT_SERVO_MAX_ANGLE, new_tilt_angle))

        current_pan_angle = final_pan_angle
        current_tilt_angle = final_tilt_angle

        # Control Servos using ServoKit
        kit.servo[PAN_SERVO_GPIO].angle = final_pan_angle
        kit.servo[TILT_SERVO_GPIO].angle = final_tilt_angle
            
        logger.info(f"Servos -> Pan: {final_pan_angle:.1f}°, Tilt: {final_tilt_angle:.1f}°")

    except Exception as e:
        logger.error(f"Error moving servos via ServoKit: {e}")


# --- WebRTC Event Handlers ---

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
            switches = data.get('sw', 0)

            logger.info(f"Received: J1_X={x}, J1_Y={y}, SW={switches}")
            
            # CONTROL THE SERVOS
            move_servos(x, y)
            
        except Exception as e:
            logger.warning(f"Error processing message: {e}")


@pc.on("track")
def on_track(track):
    logger.info(f"Receiving {track.kind} track from web app (Ignoring, but connection confirmed)")

@pc.on("connectionstatechange")
async def on_connectionstatechange():
    logger.info(f"Connection state is {pc.connectionState}")
    if pc.connectionState == "failed" or pc.connectionState == "closed":
        # Note: Closing the connection here relies on the WebSocket handler 
        # also cleanly exiting. For simple Pi control, this is sufficient.
        await pc.close() 
        logger.info("Connection failed or closed.")


async def ws_server_handler(websocket, path):
    """Handles incoming WebSocket connection for signaling."""
    global pc
    logger.info("Signaling Server: Web App connected.")

    # 1. Reset pc on new connection attempt (good practice)
    if pc.connectionState != 'closed':
        await pc.close()
    pc = RTCPeerConnection() 
    # Re-attach handlers if necessary, though decorator attaches them globally

    try:
        # 2. Receive Offer
        message = await websocket.recv()
        offer = json.loads(message)
        
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer['sdp'], type=offer['type']))
        logger.info("Offer received and set.")

        # 3. Create and Send Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Send the Answer back to the Web App via WebSocket
        await websocket.send(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
        logger.info("Answer sent to Web App.")

        # 4. Handle ICE Candidates (Relay)
        # Send the Robot's candidates to the Web App
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await websocket.send(json.dumps(candidate.to_dict()))
        
        # Loop to continuously receive ICE candidates from the Web App
        async for message in websocket:
            try:
                candidate_data = json.loads(message)
                if candidate_data.get('candidate'):
                    candidate = RTCIceCandidate(
                        candidate=candidate_data['candidate'],
                        sdpMid=candidate_data['sdpMid'],
                        sdpMLineIndex=candidate_data['sdpMLineIndex']
                    )
                    await pc.addIceCandidate(candidate)
            except Exception as e:
                logger.warning(f"Error processing received ICE message: {e}")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Signaling channel closed normally.")
    except Exception as e:
        logger.error(f"Signaling Error: {e}")
    finally:
        # Wait indefinitely until connection closes or fails
        await asyncio.Event().wait()


async def run_signaling():
    """Sets up and runs the WebSocket Signaling Server."""
    
    start_server = websockets.serve(
        ws_server_handler,
        SIGNALING_SERVER_ADDRESS,
        SIGNALING_SERVER_PORT
    )
    
    await start_server
    logger.info(f"Signaling server running on ws://{SIGNALING_SERVER_ADDRESS}:{SIGNALING_SERVER_PORT}")
    
    # Keep the main task alive until shutdown
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
        loop.run_until_complete(run_signaling())
    except KeyboardInterrupt:
        pass # Handled by the signal handler
    finally:
        logger.info("Cleaning up...")
        if servos_enabled:
            # Center servos on exit
            kit.servo[PAN_SERVO_GPIO].angle = 90
            kit.servo[TILT_SERVO_GPIO].angle = 90
            
        loop.close()
        logger.info("Shutdown complete.")