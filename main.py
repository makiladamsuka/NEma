from computervision.emotiondetection.emocapture import run_emotion_detector
from oled.emodisplay import setup_and_start_display, display_emotion


setup_and_start_display() 

while True:
    emotion = run_emotion_detector()
    print(emotion)
    
    if emotion != None:
        display_emotion(emotion)

