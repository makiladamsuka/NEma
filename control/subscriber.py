# subscriber.py
import paho.mqtt.client as mqtt
from adafruit_servokit import ServoKit
import time

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC1 = "nema/robot/controllers" # This is our P.O. Box name

def Servo_Control_Function(Servo_X, Servo_Y):

    try:
        kit = ServoKit(channels=16) 
    except ValueError:
        print("ERROR: Could not initialize ServoKit. Check I2C wiring and external power.")


    PAN_CHANNEL = 0  
    TILT_CHANNEL = 1 

    kit.servo[PAN_CHANNEL].set_pulse_width_range(min_pulse=450, max_pulse=2600)
    kit.servo[TILT_CHANNEL].set_pulse_width_range(min_pulse=450, max_pulse=2600)

    kit.servo[PAN_CHANNEL].angle = Servo_X
    kit.servo[TILT_CHANNEL].angle = Servo_Y
    print(f"Set Pan: {Servo_X}°, Tilt: {Servo_Y}°")

def map_value_for_x(value, from_min, from_max, to_min, to_max):
    mapped_value_for_x = (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    return mapped_value_for_x

def map_value_for_y(value, from_min, from_max, to_min, to_max):
    mapped_value_for_y = to_min + ((from_max - value) * (to_max - to_min) / (from_max - from_min))
    return mapped_value_for_y
# This function runs when we first connect
def on_connect(client, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}!")
    # Once connected, subscribe to our topic
    client.subscribe(MQTT_TOPIC1)
    print(f"Waiting for messages on topic: {MQTT_TOPIC1}")

# This function runs every time a new message arrives
def on_message(client, userdata, msg):
    ServoControllers = msg.payload.decode('utf-8')
    print(f"MESSAGE RECEIVED: '{ServoControllers}'")
    Servo_Axis_List = ServoControllers.split(",")
    Servo_X1 = int(Servo_Axis_List[0])
    Servo_Y1 = int(Servo_Axis_List[1])
    Servo_X2 = int(Servo_Axis_List[2])
    Servo_Y2 = int(Servo_Axis_List[3])

    mapped_servo_x = map_value_for_x(Servo_X1, 0, 1024, 50, 150)
    mapped_servo_y = map_value_for_y(Servo_Y1, 0, 1024, 30, 150)

    Servo_Control_Function(mapped_servo_x, mapped_servo_y)
    
    return_servo_values_for_face(Servo_X1, Servo_Y1)
    return_servo_values_for_wheeel(Servo_X2, Servo_Y2)

def return_servo_values_for_face(Servo_X1, Servo_Y1):
    return Servo_X1, Servo_Y1

def return_servo_values_for_wheeel(Servo_X2, Servo_Y2):
    return Servo_X2, Servo_Y2

# --- Setup ---
client = mqtt.Client()
client.on_connect = on_connect # Attach our function
client.on_message = on_message # Attach our other function

print("Connecting to broker...")
client.connect(MQTT_BROKER, 1883, 60)

# This is a blocking loop that keeps the script
# running and listening for messages forever.
client.loop_forever()