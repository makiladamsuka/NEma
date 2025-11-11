# subscriber.py
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC1 = "nema/robot/control_x"
MQTT_TOPIC2 = "nema/robot/control_x" # This is our P.O. Box name

# This function runs when we first connect
def on_connect1(client, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}!")
    # Once connected, subscribe to our topic
    client.subscribe(MQTT_TOPIC1)
    print(f"Waiting for messages on topic: {MQTT_TOPIC1}")

def on_connect2(client, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}!")
    # Once connected, subscribe to our topic
    client.subscribe(MQTT_TOPIC2)
    print(f"Waiting for messages on topic: {MQTT_TOPIC2}")


# This function runs every time a new message arrives
def on_message1(client, userdata, msg):
    print(f"MESSAGE RECEIVED: '{msg.payload.decode('utf-8')}'")

def on_message2(client2, userdata, msg):
    print(f"MESSAGE RECEIVED: '{msg.payload.decode('utf-8')}'")

# --- Setup ---
client1 = mqtt.Client()
client1.on_connect = on_connect1 # Attach our function
client1.on_message = on_message1 # Attach our other function
client2 = mqtt.Client()
client2.on_connect = on_connect2 # Attach our function
client2.on_message = on_message2

print("Connecting to broker...")
client1.connect(MQTT_BROKER, 1883, 60)
client2.connect(MQTT_BROKER, 1883, 60)

# This is a blocking loop that keeps the script
# running and listening for messages forever.
client1.loop_forever()
client2.loop_forever()