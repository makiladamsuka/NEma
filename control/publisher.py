# publisher_loop.py
import paho.mqtt.client as mqtt
import time
import random # Import the random library

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "nema/robot/test" # Must be the exact same topic!

# --- Setup ---
client = mqtt.Client()

print(f"Connecting to broker {MQTT_BROKER}...")
try:
    client.connect(MQTT_BROKER, 1883, 60)
    print("Connected!")
except Exception as e:
    print(f"Could not connect: {e}")
    exit(1)

# Start the client's network loop in the background.
# This handles sending messages and auto-reconnecting.
client.loop_start()

try:
    # Start an endless loop
    while True:
        # 1. Generate a random number
        random_num = random.randint(0, 100)
        
        # 2. Format the number as a string to send it
        message = str(random_num)
        
        # 3. Publish (send) the message
        print(f"Sending message: '{message}'")
        client.publish(MQTT_TOPIC, message)
        
        # 4. Wait for 1 second before sending the next one
        time.sleep(0.1)

except KeyboardInterrupt:
    # This runs if you press Ctrl+C
    print("Stopping the loop...")

finally:
    # Clean up
    print("Disconnecting from broker.")
    client.loop_stop() # Stop the background loop
    client.disconnect()


