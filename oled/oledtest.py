from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306 
from luma.core.render import canvas # <-- ADDED for drawing text
import time

# --- CONFIGURATION ---
# NOTE: You MUST check the correct I2C addresses for your two OLED screens.
I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3c  # Address for the Left Screen
RIGHT_OLED_ADDRESS = 0x3d # Address for the Right Screen (Change this if needed)
# ---------------------

def setup_device(address, name):
    """Initializes a single SSD1306 device."""
    serial = i2c(port=I2C_PORT, address=address)
    try:
        # Use ssd1306 for a typical 128x64 display 
        device = ssd1306(serial)
        print(f"SUCCESS: {name} OLED device (Address: {hex(address)}) initialized.")
        return device
    except Exception as e:
        print(f"ERROR: Could not connect to {name} device (Address: {hex(address)}). Check power/wiring.")
        print(f"Detail: {e}")
        return None

def run_connection_test():
    """Initializes both devices and displays test text on each."""
    print("--- Starting Dual OLED Connection Test ---")
    
    # 1. Setup both connections
    left_device = setup_device(LEFT_OLED_ADDRESS, "LEFT")
    right_device = setup_device(RIGHT_OLED_ADDRESS, "RIGHT")

    if not left_device and not right_device:
        print("FATAL: Neither device could be connected. Test aborted.")
        return

    # 2. Draw simple text on connected devices
    try:
        # Use the canvas context manager to safely draw on the screen
        if left_device:
            with canvas(left_device) as draw:
                # Draw text at the top left corner (0, 0)
                draw.text((0, 0), "LEFT SCREEN", fill="white")
                draw.text((0, 10), "TEST OK", fill="white")
            print("Displayed test message on LEFT screen.")

        if right_device:
            with canvas(right_device) as draw:
                # Draw text at the top left corner (0, 0)
                draw.text((0, 0), "RIGHT SCREEN", fill="white")
                draw.text((0, 10), "TEST OK", fill="white")
            print("Displayed test message on RIGHT screen.")
            
        if left_device or right_device:
            print("Waiting 5 seconds to view test messages...")
            time.sleep(5)
            
    except Exception as e:
        print(f"An error occurred during text drawing: {e}")

    finally:
        # 3. Always clear the screens
        if left_device:
            left_device.clear() 
        if right_device:
            right_device.clear()
        print("Screens cleared. Test complete.")

if __name__ == "__main__":
    run_connection_test()
