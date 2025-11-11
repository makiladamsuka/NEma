# subscriber.py
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC1 = "nema/robot/controllers" # This is our P.O. Box name

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
    Servo_X1 = Servo_Axis_List[0]
    Servo_Y1 = Servo_Axis_List[1]
    Servo_X2 = Servo_Axis_List[2]
    Servo_Y2 = Servo_Axis_List[3]
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