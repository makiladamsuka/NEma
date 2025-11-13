import asyncio
import cv2
import json
import logging
import time
import av 
import platform

# --- FIX 1: Explicitly import RTCIceCandidate to resolve 'name not defined' error ---
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

# Set up logging for aiortc
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("robot_test")
logger.setLevel(logging.INFO)

# This queue will hold the video frames we receive from the browser
remote_video_queue = asyncio.Queue()

class OpenCVTrack(VideoStreamTrack):
    """
    A video stream track that captures frames from a local webcam using OpenCV.
    """
    def __init__(self):
        super().__init__()
        logger.info("Initializing OpenCVTrack...")
        # Note: In a real Pi deployment, the index might be different (e.g., 2 or -1)
        self.cap = cv2.VideoCapture(0)  
        if not self.cap.isOpened():
            logger.error("Could not open camera 0")
            raise Exception("Could not open camera 0")
        self.relay = MediaRelay()

    def __del__(self):
        """Clean up the camera resource."""
        if self.cap.isOpened():
            self.cap.release()

    async def recv(self):
        """
        Called by aiortc when it needs a new video frame.
        """
        pts, time_base = await self.next_timestamp()

        ret, img = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            # Loop a blank frame if read fails
            img = av.VideoFrame(640, 480, "bgr24").to_ndarray()

        # Convert the OpenCV BGR image to an av.VideoFrame
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        
        return frame

async def main():
    """
    The main asyncio function to run the WebRTC client.
    """
    pc = RTCPeerConnection()

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"DataChannel '{channel.label}' created")

        @channel.on("message")
        def on_message(message):
            try:
                # Parse the JSON control packet from the browser
                data = json.loads(message)
                logger.info(f"Controls Received: {data}")
            except Exception as e:
                logger.warning(f"Failed to parse data channel message: {e}")

    @pc.on("track")
    async def on_track(track):
        logger.info(f"Receiving {track.kind} track from browser")
        if track.kind == "video":
            # Start a background task to pull frames from the track
            asyncio.create_task(video_receiver(track))
        
    @pc.on("icecandidate")
    def on_icecandidate(event):
        if event.candidate:
            print("\n--- Robot's ICE Candidate (Copy to Browser) ---")
            print(json.dumps(event.candidate.to_dict()))
            print("-------------------------------------------------")

    # 1. Add our local webcam track to send to the browser
    try:
        local_video = OpenCVTrack()
        pc.addTrack(local_video)
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        return

    # --- Manual Signaling Process ---

    # 2. Get Offer from Browser
    print("\n--- Step 1: Paste 'Web App Offer' from browser ---")
    print("(Paste the full JSON and press Enter)")
    offer_json = input("Offer: ")
    try:
        offer_dict = json.loads(offer_json)
        offer_sdp = RTCSessionDescription(sdp=offer_dict["sdp"], type=offer_dict["type"])
        await pc.setRemoteDescription(offer_sdp)
    except Exception as e:
        logger.error(f"Failed to parse Offer: {e}")
        return

    # 3. Create and Print Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    print("\n--- Step 2: Copy 'Robot's Answer' to Browser ---")
    print(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
    print("--------------------------------------------------")

    # 4. Get Web App ICE Candidates
    print("\n--- Step 3: Paste 'Web App ICEs' from browser ---")
    print("(Paste one full JSON candidate at a time, then a blank line to finish)")
    while True:
        candidate_json = input("Web App ICE: ")
        if candidate_json.strip() == "":
            logger.info("Finished adding Web App ICE candidates.")
            break
        try:
            candidate_dict = json.loads(candidate_json)
            
            # --- FIX 2: Correct RTCIceCandidate construction ---
            candidate = RTCIceCandidate(
                candidate=candidate_dict["candidate"],
                sdpMid=candidate_dict["sdpMid"],
                sdpMLineIndex=candidate_dict["sdpMLineIndex"]
            )

            await pc.addIceCandidate(candidate)
        except Exception as e:
            logger.warning("Failed to add ICE candidate: %s", e) 

    logger.info("Connection setup complete. Running video loop...")

    # --- Main Application Loop ---
    # This loop will display the video received from the browser
    cv2.namedWindow("Remote Video (from Browser)", cv2.WINDOW_AUTOSIZE)
    
    while True:
        try:
            # FIX 3: Use get_nowait() to prevent blocking the main loop 
            # (which helps prevent the 'Future attached to a different loop' error)
            remote_img = remote_video_queue.get_nowait()
            cv2.imshow("Remote Video (from Browser)", remote_img)

        except asyncio.QueueEmpty:
            # This is expected when no new frame has arrived yet
            pass
        except Exception as e:
            # Catch potential OpenCV or display errors
            logger.warning(f"Error displaying remote frame: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Yield control to other asyncio tasks
        await asyncio.sleep(0.001) 

    logger.info("Shutting down...")
    await pc.close()
    # The local_video destructor (__del__) handles cap.release()
    cv2.destroyAllWindows()

async def video_receiver(track):
    """
    Receives frames from a remote video track and puts them in the queue.
    """
    while True:
        try:
            frame = await track.recv()
            # Convert the av.VideoFrame to an OpenCV (numpy) image
            img = frame.to_ndarray(format="bgr24")
            await remote_video_queue.put(img)
        except Exception as e:
            logger.warning(f"Error receiving remote frame: {e}")
            return

if __name__ == "__main__":
    # --- FIX 4: Explicitly set Windows Selector Event Loop Policy ---
    # This directly addresses the "Future attached to a different loop" RuntimeError.
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # ------------------------------------------------------------------
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")