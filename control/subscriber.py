# subscriber.py
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC1 = "nema/robot/control_x"
MQTT_TOPIC2 = "nema/robot/control_x" # This is our P.O. Box name

# This function runs when we first connect
def on_connect(client1, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}!")
    # Once connected, subscribe to our topic
    client1.subscribe(MQTT_TOPIC1)
    print(f"Waiting for messages on topic: {MQTT_TOPIC1}")

def on_connect(client2, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}!")
    # Once connected, subscribe to our topic
    client2.subscribe(MQTT_TOPIC2)
    print(f"Waiting for messages on topic: {MQTT_TOPIC2}")


# This function runs every time a new message arrives
def on_message(client1, userdata, msg1):
    print(f"MESSAGE RECEIVED: '{msg1.payload.decode('utf-8')}'")

def on_message(client2, userdata, msg2):
    print(f"MESSAGE RECEIVED: '{msg2.payload.decode('utf-8')}'")

# --- Setup ---
client1 = mqtt.Client()
client1.on_connect = on_connect # Attach our function
client1.on_message = on_message # Attach our other function
client2 = mqtt.Client()
client2.on_connect = on_connect # Attach our function
client2.on_message = on_message

print("Connecting to broker...")
client1.connect(MQTT_BROKER, 1883, 60)
client2.connect(MQTT_BROKER, 1883, 60)

# This is a blocking loop that keeps the script
# running and listening for messages forever.
client1.loop_forever()
client2.loop_forever()