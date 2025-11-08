from adafruit_servokit import ServoKit
import time

try:
    kit = ServoKit(channels=16) 
except ValueError:
    print("ERROR: Could not initialize ServoKit. Check I2C wiring and external power.")


PAN_CHANNEL = 0  
TILT_CHANNEL = 1 

kit.servo[PAN_CHANNEL].set_pulse_width_range(min_pulse=450, max_pulse=2600)
kit.servo[TILT_CHANNEL].set_pulse_width_range(min_pulse=450, max_pulse=2600)

def move_servos(pan_angle, tilt_angle):
    kit.servo[PAN_CHANNEL].angle = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle
    print(f"Set Pan: {pan_angle}°, Tilt: {tilt_angle}°")


def main():
    while True:
        user_input = input("Enter pan angle and tilt angle (e.g., 50-150 30-150): ")
        pan_str, tilt_str = user_input.split()

        pan_angle = int(pan_str)
        tilt_angle = int(tilt_str)
        
        move_servos(pan_angle, tilt_angle)


if __name__ == '__main__':
    main()