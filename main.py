print("INITIALIZING DISPLAY")
from oled.emodisplay import setup_and_start_display, display_emotion
print("INITIALIZING DETECTION MODEL")
from computervision.emotiondetection.emocapture import run_emotion_detector
import time 

print("**********SUCCESS******************")


setup_and_start_display() 


while True:
    emotion, face_coordinates = run_emotion_detector()
    print(f"Detected: {emotion} coordinates: {face_coordinates}")
    
    if emotion is not None:      
        print(f"-> Displaying **{emotion}**. Pausing detection...")
        
        # 3. Queue the emotion
        display_emotion(emotion) 
        
        # 4. Wait for the animation to (presumably) finish.
        # This is a simple, synchronous way to pause the detection loop.
        time.sleep(5.0) # Pause detection for 1 second (adjust as needed for animation length)
        print("-> Detection resumed.")