# signaling_server.py
import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signaling_server")

# Store the connected clients
WEB_CLIENT = None
PI_CLIENT = None

async def handler(websocket, path):
    global WEB_CLIENT, PI_CLIENT
    
    # Register clients based on the path they connect to
    if path == "/web":
        logger.info("Web App client connected")
        WEB_CLIENT = websocket
    elif path == "/pi":
        logger.info("Raspberry Pi client connected")
        PI_CLIENT = websocket
    else:
        logger.warning(f"Unknown client connected from path: {path}")
        return

    try:
        # Listen for messages
        async for message in websocket:
            # When web client sends a message, forward it to pi
            if websocket is WEB_CLIENT and PI_CLIENT:
                logger.info("Message from Web -> Pi")
                await PI_CLIENT.send(message)
            # When pi client sends a message, forward it to web
            elif websocket is PI_CLIENT and WEB_CLIENT:
                logger.info("Message from Pi -> Web")
                await WEB_CLIENT.send(message)
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        # Unregister clients on disconnect
        if websocket is WEB_CLIENT:
            WEB_CLIENT = None
            logger.info("Web App client disconnected")
        elif websocket is PI_CLIENT:
            PI_CLIENT = None
            logger.info("Raspberry Pi client disconnected")

async def main():
    logger.info("Starting signaling server on ws://0.0.0.0:8765")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())