/*
 * ESP32 Robot Head Controller
 * * Reads two joysticks and two switches, then sends the data
 * over Bluetooth Low Energy (BLE) to a web application.
 * * IMPORTANT: This code is for an ESP32, NOT an ESP8266.
 * You need an ESP32 for its multiple analog pins and BLE server capabilities.
 *
 * WIRING (ESP32):
 * - Joy1 X  -> GPIO 34 (ADC1_CH6)
 * - Joy1 Y  -> GPIO 35 (ADC1_CH7)
 * - Joy2 X  -> GPIO 32 (ADC1_CH4)
 * - Joy2 Y  -> GPIO 33 (ADC1_CH5)
 * - Switch 1 -> GPIO 25 (and GND)
 * - Switch 2 -> GPIO 26 (and GND)
 *
 * (All joysticks also need 3.3V and GND)
 */

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h> // For BLE Notifications

// --- Pin Definitions ---
// Use any ADC1 pins for joysticks (e.g., 32, 33, 34, 35, 36, 39)
#define JOY1_X_PIN 34
#define JOY1_Y_PIN 35
#define JOY2_X_PIN 32
#define JOY2_Y_PIN 33

// Use any digital pins for switches
#define SW1_PIN 25
#define SW2_PIN 26

// --- BLE UUIDs ---
// These MUST match the UUIDs in your index.html file.
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Data structure to hold all control data
// This is what we will send over BLE.
struct ControlData {
  int8_t  joy1_x; // -127 to 127
  int8_t  joy1_y; // -127 to 127
  int8_t  joy2_x; // -127 to 127
  int8_t  joy2_y; // -127 to 127
  uint8_t switches; // 1 byte, 8 bits for 8 switches
};


// Global BLE Characteristic
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
ControlData lastControllerInput; // To send data only on change


ControlData controllerInput;

// --- Helper Functions ---

// Maps and clamps joystick analog input (0-4095) to a signed byte (-127 to 127)
// Includes a "deadzone" to prevent drift when joystick is centered.
int8_t mapJoystick(int value) {
  // Map 0-4095 to -127 to 127
  int mapped = map(value, 0, 4095, -127, 127);
  
  // Clamp to be safe
  return constrain(mapped, -127, 127);
}

// Reads all inputs and updates the global 'controllerInput' struct
void readInputs() {
  controllerInput.joy1_x = mapJoystick(analogRead(JOY1_X_PIN));
  // Invert Y-axis for standard "up = positive" control
  controllerInput.joy1_y = mapJoystick(analogRead(JOY1_Y_PIN)); 
  
  controllerInput.joy2_x = mapJoystick(analogRead(JOY2_X_PIN));
  controllerInput.joy2_y = -mapJoystick(analogRead(JOY2_Y_PIN));

  // Read switches (using INPUT_PULLUP, so LOW means "pressed")
  bool sw1_pressed = !digitalRead(SW1_PIN);
  bool sw2_pressed = !digitalRead(SW2_PIN);

  // Pack switch data into the 'switches' byte
  controllerInput.switches = 0; // Clear all bits
  if (sw1_pressed) {
    controllerInput.switches |= (1 << 0); // Set bit 0
  }
  if (sw2_pressed) {
    controllerInput.switches |= (1 << 1); // Set bit 1
  }
  // You can add 6 more switches here (bit 2 to 7)
}

// --- BLE Server Callbacks ---
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Device connected");
    }

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Device disconnected - restarting advertising");
      pServer->getAdvertising()->start(); // Restart advertising
    }
};

// --- Setup ---
void setup() {
  Serial.begin(115200);
  Serial.println("Starting Robot Controller...");

  // Initialize switch pins with internal pull-up resistors
  pinMode(SW1_PIN, INPUT_PULLUP);
  pinMode(SW2_PIN, INPUT_PULLUP);
  
  // Set ADC width for better precision (optional but good)
  analogReadResolution(12); // 0-4095

  // --- Create the BLE Server ---
  BLEDevice::init("ESP32 Robot Controller"); // Set device name
  
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks()); // Set connect/disconnect callbacks

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create the characteristic
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  // Add a 2902 descriptor
  // This is required for the client (web app) to enable notifications
  pCharacteristic->addDescriptor(new BLE2902());

  pService->start(); // Start the service

  // --- Start Advertising ---
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->start();
  
  Serial.println("Waiting for a client connection...");
  
  // Initialize lastControllerInput
  memset(&lastControllerInput, 0, sizeof(ControlData));
}

// --- Loop ---
void loop() {
  
  // Check if a client (phone) is connected
  if (deviceConnected) {
    
    // Read the latest data from joysticks and switches
    readInputs();

    // Check if data has actually changed
    if (memcmp(&controllerInput, &lastControllerInput, sizeof(ControlData)) != 0) {
      
      // Set the characteristic's value
      pCharacteristic->setValue((uint8_t*)&controllerInput, sizeof(controllerInput));
      
      // Send a notification to the client
      pCharacteristic->notify();
      
      // Update last sent data
      memcpy(&lastControllerInput, &controllerInput, sizeof(ControlData));

      // Optional: Print to serial monitor for debugging
      // Serial.printf("J1: %d, %d | J2: %d, %d | SW: %d\n",
      //   controllerInput.joy1_x, controllerInput.joy1_y,
      //   controllerInput.joy2_x, controllerInput.joy2_y,
      //   controllerInput.switches);
    }
  }
  
  // Send updates 50 times per second (20ms delay)
  // This gives a very responsive feel.
  delay(20); 
}