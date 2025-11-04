import os
import base64
import time
import cv2
import io
from PIL import Image
from openai import OpenAI
# Removed dotenv and specific OpenRouter imports/configs as they are no longer needed

# --- ⚙️ CONFIGURATION FOR OLLAMA (MOONDREAM:V2) ---
# The model name you have pulled via Ollama (e.g., 'ollama pull moondream:v2')
MODEL_NAME = "moondream:v2" 
# The default local endpoint for Ollama's OpenAI-compatible API
OLLAMA_BASE_URL = "http://localhost:11434/v1" 

# --- 🎯 LIVE WEBCAM CONFIGURATION ---
# 1. Input: The index of the webcam to use (0 is usually the default internal camera)
WEBCAM_INDEX = 0 
# 2. Output: File to save the analysis log
OUTPUT_LOG_FILE = "moondream_webcam_analysis_log.txt" 
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
    image_pil.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# --- Main Function: Send Request to Ollama ---
def ask_llm_with_ollama(base64_image, question, model_name, base_url):
    """Sends the base64 image data and question to the local Ollama API."""
    try:
        # Initialize the OpenAI client, pointing it to the local Ollama server
        client = OpenAI(
            base_url=base_url,
            # Ollama does not require an actual key, but the client often expects one
            api_key='ollama-is-local', 
        )
        
        # Construct the multimodal content for the chat API
        messages_content = [
            {"type": "text", "text": question},
            # The image must be passed as a data URI
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
        ]

        # Use the chat completions endpoint
        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': messages_content}],
            max_tokens=100, # Moondream is small, keep max_tokens low for fast responses
            temperature=0.2 # Lower temperature for descriptive, factual tasks
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Provide helpful instructions if the connection fails
        return (f"API Error: Could not connect to Ollama at {base_url}. "
                f"Ensure Ollama is running, the model '{model_name}' is pulled, and the port (11434) is correct. Error details: {e}")

# --- Execution: Webcam Live Loop ---
if __name__ == "__main__":
    # Ensure the model is available before starting
    try:
        print(f"✅ Starting LIVE webcam analysis. Model: {MODEL_NAME}")
        print(f"  Ollama Base URL: {OLLAMA_BASE_URL}")
        print(f"  Analysis saved to: {OUTPUT_LOG_FILE}")
        
        # 1. Open the webcam (using WEBCAM_INDEX, typically 0)
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open webcam at index {WEBCAM_INDEX}. Check if the camera is connected or in use.")
            exit()
        
        frame_count = 0
        analyzed_frame_count = 0
        # Tracks the time of the last successful analysis call
        last_analysis_time = time.time()

        # Open the log file for writing
        with open(OUTPUT_LOG_FILE, 'w') as log_file:
            log_file.write(f"--- Live Webcam Analysis Log ---\n")
            log_file.write(f"Model: {MODEL_NAME}, Interval: {ANALYSIS_INTERVAL_SECONDS} seconds\n\n")

            # 2. Main live video loop
            while True:
                ret, frame = cap.read()
                
                # Break if no frame is captured (e.g., camera unplugged)
                if not ret:
                    print("Camera stream ended or failed.")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # 3. Time-based check to avoid sending too many requests to Ollama
                if current_time - last_analysis_time >= ANALYSIS_INTERVAL_SECONDS:
                    last_analysis_time = current_time
                    analyzed_frame_count += 1
                    
                    print(f"⏱️ Analyzing frame {frame_count} (Time elapsed: {current_time - last_analysis_time:.2f}s)...", end='\r')
                    
                    # Encode and send to LLM
                    image_data = encode_frame_to_base64(frame)
                    llm_answer = ask_llm_with_ollama(
                        image_data, 
                        VIDEO_QUESTION, 
                        MODEL_NAME, 
                        OLLAMA_BASE_URL
                    )
                    
                    end_time = time.time()
                    
                    log_entry = f"[Frame: {frame_count}, Latency: {end_time - current_time:.2f}s] -> {llm_answer}\n"
                    log_file.write(log_entry)
                    
                    # Use strip() to clean up the output description
                    print(f"✅ Analysis at {current_time:.2f}s: {llm_answer.strip()}")
                
                # Display the live feed
                cv2.imshow('Live Moondream Analysis (Press Q to quit)', frame)

                # Check for 'q' keypress to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print(f"\n\nWebcam analysis complete. Total frames analyzed: {analyzed_frame_count}")
        cap.release()
        cv2.destroyAllWindows() # Close the OpenCV window
        
    except NameError as e:
        print(f"\n\n🔴 ERROR: Missing import or configuration. Please ensure you have installed the required libraries.")
        print("Run: pip install openai opencv-python Pillow")
        print(f"Detailed error: {e}")
    except Exception as e:
        print(f"\n\n🔴 A general error occurred during execution: {e}")
