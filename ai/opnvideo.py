import os
import base64
import time
import cv2 # <--- NEW: For webcam capture
import io
from PIL import Image # <--- NEW: For easy image-to-bytes conversion
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl:free"
# The question you want the model to answer about the live frame
VIDEO_QUESTION = "Describe the main action, object, and person in this frame in one short sentence."
# How often (in seconds) to send a frame to the model
ANALYSIS_INTERVAL_SECONDS = 5
# Index of your camera (usually 0 for the default webcam)
CAMERA_INDEX = 0

# 1. --- Helper Function: Encode OpenCV Frame to Base64 ---
# This replaces your old 'encode_image' which read from a file.
def encode_frame_to_base64(frame):
    """Converts an OpenCV frame (NumPy array) into a base64 encoded string."""
    # Convert OpenCV BGR frame to PIL RGB format
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Save the PIL image to a bytes buffer as JPEG
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG") 
    # Base64 encode the bytes and decode to a string
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# 2. --- Main Function: Send Request to OpenRouter ---
def ask_llm_with_openrouter(base64_image, question, model_name, base_url, api_key):
    """Sends the base64 image data and question to the OpenRouter API."""
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        
        # Multimodal message content using a data URL for the image
        messages_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': messages_content}],
            max_tokens=200, # Keep the response short for quick analysis
            temperature=0.4
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"API Error: {e}"

# 3. --- Execution: Live Loop ---
if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("🔴 ERROR: OPENROUTER_API_KEY environment variable not set. Please check your .env file.")
        exit()

    print(f"✅ Starting real-time video analysis. Model: {MODEL_NAME}")
    print(f"  Analyzing a new frame every {ANALYSIS_INTERVAL_SECONDS} seconds.")
    print("  Press 'q' key while the video window is open to stop.")
    
    # 3.1. Start the camera feed
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Error: Could not open camera. Check CAMERA_INDEX.")
        exit()

    start_time = time.time()
    llm_answer = "Waiting for first analysis..." 
    
    # 3.2. Main video loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Video stream ended or error reading frame.")
            break
        
        current_time = time.time()
        
        # 3.3. Check if it's time to send a frame to the model
        if (current_time - start_time) >= ANALYSIS_INTERVAL_SECONDS:
            
            image_data = encode_frame_to_base64(frame)
            llm_answer = ask_llm_with_openrouter(
                image_data, 
                VIDEO_QUESTION, 
                MODEL_NAME, 
                BASE_URL, 
                OPENROUTER_API_KEY
            )
            
            print(f"\n🧠 Nemotron Answer: {llm_answer}")
            
            # Reset the timer
            start_time = current_time

        # 3.4. Display the live feed and the answer
        
        # Prepare text for display
        display_text = f"ANALYSIS: {llm_answer}"
        # Draw the text onto the frame (position at the bottom of the window)
        cv2.putText(frame, 
                    display_text, 
                    (10, frame.shape[0] - 10), # Position
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, # Size
                    (0, 255, 0), # Green color (BGR format)
                    2, # Thickness
                    cv2.LINE_AA)

        cv2.imshow('Live Nemotron Analysis (Press Q to quit)', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 3.5. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo analysis stopped.")