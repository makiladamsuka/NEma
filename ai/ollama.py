import os
import base64
import time
import cv2
import io
from PIL import Image
import requests # New import for local API calls
from dotenv import load_dotenv

# --- Configuration ---
# Note: Since Ollama runs locally, API key environments are removed.
load_dotenv()

# --- 🎯 NEW CONFIGURATION FOR OLLAMA ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream" # Ensure this model is pulled in your local Ollama instance
# -------------------------------------

# 1. Input: The video source. For a webcam, use the integer index (0 is usually the default camera).
# For a video file, you would use a string like "my_video.mp4".
VIDEO_INPUT_SOURCE = 0 
# 2. Output: File to save the analysis log
OUTPUT_LOG_FILE = "webcam_analysis_log.txt" 
# 3. Analysis: The question you want the model to answer about each frame
VIDEO_QUESTION = "Describe the main action, object, and person in this frame in one short sentence."
# 4. Analysis: How often (in seconds) to analyze a frame
ANALYSIS_INTERVAL_SECONDS = 5 

# --- Helper Function: Encode OpenCV Frame to Base64 ---
def encode_frame_to_base64(frame):
    """Converts an OpenCV frame (NumPy array) into a base64 encoded string."""
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    # Using 'JPEG' format for smaller data size and faster transfer
    # NOTE: Moondream often prefers PNG or JPG. We stick with JPEG.
    image_pil.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# --- Main Function: Send Request to Ollama ---
def ask_llm_with_ollama(base64_image, question, model_name, ollama_url):
    """Sends the base64 image data and question to the local Ollama API."""
    try:
        # Ollama API payload structure for multimodal generation
        payload = {
            "model": model_name,
            "prompt": question,
            "images": [base64_image], # Base64 data goes here
            "stream": False,         # Use non-streaming mode
            "options": {
                "temperature": 0.4,
                "num_predict": 200   # Equivalent to max_tokens
            }
        }

        # Make the synchronous POST request
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Ollama response is a JSON object with the result in the 'response' field
        return response.json().get("response", "API Error: No response text found.")

    except requests.exceptions.ConnectionError:
        return f"API Error: Could not connect to Ollama at {ollama_url}. Is Ollama running locally?"
    except requests.exceptions.RequestException as e:
        return f"API Error during request: {e}"
    except Exception as e:
        return f"Unexpected Error: {e}"

# --- Execution: Video File Loop ---
if __name__ == "__main__":
    SOURCE_DISPLAY_NAME = "Webcam Feed (Index 0)"

    print(f"✅ Starting live analysis of: '{SOURCE_DISPLAY_NAME}'. Model: {MODEL_NAME}")
    print(f"  Ollama Endpoint: {OLLAMA_URL}")
    print(f"  Analysis saved to: {OUTPUT_LOG_FILE}")
    
    # 1. Open the video source (webcam index 0)
    cap = cv2.VideoCapture(VIDEO_INPUT_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {VIDEO_INPUT_SOURCE}. Check if your webcam is connected or in use by another application.")
        exit()
    
    # Get important video properties
    fps = cap.get(cv2.CAP_PROP_FPS) # Frames per second
    if fps == 0:
        print("⚠️ Warning: Could not read webcam FPS. Assuming 30.")
        fps = 30
        
    # Calculate how many frames to skip based on the desired interval
    frames_to_skip = int(fps * ANALYSIS_INTERVAL_SECONDS)
    frame_count = 0
    analyzed_frame_count = 0

    # Open the log file for writing
    with open(OUTPUT_LOG_FILE, 'w') as log_file:
        log_file.write(f"--- Webcam Analysis Log ---\n")
        log_file.write(f"Model: {MODEL_NAME}, Interval: {ANALYSIS_INTERVAL_SECONDS} seconds\n")
        log_file.write(f"Question: {VIDEO_QUESTION}\n\n")

        # 2. Main video loop
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Since it's a live feed, we don't break on not ret unless there's a serious error
            if not ret:
                print("❌ Fatal Error: Lost connection to webcam stream.")
                break
            
            # 3. Check if it's an analysis frame based on the frame count
            if frame_count % frames_to_skip == 0:
                analyzed_frame_count += 1
                
                # Calculate the time in the video
                video_time = frame_count / fps
                
                print(f"⏱️ Analyzing frame {frame_count} ({video_time:.2f}s)...", end='\r')
                
                # Encode and send to LLM
                image_data = encode_frame_to_base64(frame)
                llm_answer = ask_llm_with_ollama(
                    image_data, 
                    VIDEO_QUESTION, 
                    MODEL_NAME, 
                    OLLAMA_URL
                )
                
                log_entry = f"[Time: {video_time:.2f}s, Frame: {frame_count}] -> {llm_answer}"
                
                # Write to file and print to console
                log_file.write(log_entry + "\n")
                print(f"✅ Analysis at {video_time:.2f}s: {llm_answer}")
            
            frame_count += 1
            
    # Release the video capture object
    cap.release()
    print("\n\n✅ Webcam analysis session ended.")
    print(f"Total frames analyzed: {analyzed_frame_count}")
    print(f"Results saved to {OUTPUT_LOG_FILE}")
