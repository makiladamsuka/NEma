# subscriber.py
import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "nema/robot/test" # This is our P.O. Box name

# This function runs when we first connect
def on_connect(client, userdata, flags, rc):
    print(f"Connected to {MQTT_BROKER}!")
    # Once connected, subscribe to our topic
    client.subscribe(MQTT_TOPIC)
    print(f"Waiting for messages on topic: {MQTT_TOPIC}")

# This function runs every time a new message arrives
def on_message(client, userdata, msg):
    print(f"MESSAGE RECEIVED: '{msg.payload.decode('utf-8')}'")

# --- Setup ---
client = mqtt.Client()
client.on_connect = on_connect # Attach our function
client.on_message = on_message # Attach our other function

print("Connecting to broker...")
client.connect(MQTT_BROKER, 1883, 60)

# This is a blocking loop that keeps the script
# running and listening for messages forever.
client.loop_forever()