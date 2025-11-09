import cv2
from emocapturev2 import run_emotion_detector


while True:


    emotion, coords = run_emotion_detector()
    print(emotion, coords)

