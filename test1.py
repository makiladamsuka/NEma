print("INITIALIZING DISPLAY")
from oled.emodisplay import setup_and_start_display, display_emotion

import time 


print("**********SUCCESS******************")


setup_and_start_display() 


while True:
    user_input = input("enter emotion")
    display_emotion(user_input)