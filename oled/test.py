from emodisplay import setup_and_start_display, display_emotion
import time

setup_and_start_display() 

while True:
    user_in = input("Enter emotion: ")
    
    display_emotion(user_in)