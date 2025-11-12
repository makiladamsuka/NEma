import cv2
import numpy as np
import tensorflow as tf
import os
import time
from picamera2 import Picamera2
from simple_pid import PID
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import sys 

FRAME_WIDTH, FRAME_HEIGHT = 640, 480 # Using 640x480 for better speed on the Pi


I2C_ADDRESS = 0x40
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_CENTER = 90
TILT_CENTER = 90

PAN_Kp, PAN_Ki, PAN_Kd = 4/10, .001, 9/10
TILT_Kp, TILT_Ki, TILT_Kd = 4/10, .001, 9/10

SMOOTHING_FACTOR = 0.01
PID_MAX_OFFSET = 60

MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/media2.tflite'
YUNET_MODEL_PATH = '/home/nema/Documents/NEma/computervision/emotiondetection/face_detection_yunet_2023mar.onnx' 
YUNET_INPUT_SIZE = (320, 320) 
EMOTION_LABELS = ['Happy','Smile']
CONFIDENCE_THRESHOLD = 0.50

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite emotion model: {e}") 
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'] 
TFLITE_INPUT_H, TFLITE_INPUT_W = INPUT_SHAPE[1], INPUT_SHAPE[2]

try:
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, 
        "", 
        YUNET_INPUT_SIZE, 
        0.4,
        0.3, 
        5000 
    )
    print("YuNet Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading YuNet model: {e}") 
    sys.exit(1)



try:
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = 50 


    pan_servo = servo.Servo(pca.channels[PAN_CHANNEL], min_pulse=500, max_pulse=2500)
    tilt_servo = servo.Servo(pca.channels[TILT_CHANNEL], min_pulse=500, max_pulse=2500)


    pan_servo.angle = PAN_CENTER
    tilt_servo.angle = TILT_CENTER
    current_pan_angle = PAN_CENTER
    current_tilt_angle = TILT_CENTER

    print(f"PCA9685 initialized. Servos set to {PAN_CENTER}Â°, {TILT_CENTER}Â° tilt.")

except ValueError:
    print("Error: Could not find PCA9685 at the specified I2C address.")
    sys.exit(1)
except ImportError as e:
    print(f"Error: Required library not found ({e}). Ensure adafruit-blinka and adafruit-circuitpython-pca9685 are installed.")
    sys.exit(1)


PAN_SETPOINT = FRAME_WIDTH / 2
TILT_SETPOINT = FRAME_HEIGHT / 2

pan_pid = PID(PAN_Kp, PAN_Ki, PAN_Kd, setpoint=PAN_SETPOINT)
pan_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)

tilt_pid = PID(TILT_Kp, TILT_Ki, TILT_Kd, setpoint=TILT_SETPOINT)
tilt_pid.output_limits = (-PID_MAX_OFFSET, PID_MAX_OFFSET)

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"} 
)

picam2.configure(config)
picam2.start()
time.sleep(1.0) 
print(f"Picamera2 started at {FRAME_WIDTH}x{FRAME_HEIGHT} in BGR888.")



def cleanup_and_exit():
    """Stops the camera, resets servos, and closes OpenCV windows."""
    print("\nStopping camera and resetting servos...")
    picam2.stop()
    cv2.destroyAllWindows()
    pan_servo.angle = PAN_CENTER 
    tilt_servo.angle = TILT_CENTER
    # It takes a moment for the servo to move before the script exits
    time.sleep(0.5) 
    sys.exit(0)

try:
    while True:
        frame = picam2.capture_array()
        
        face_detector.setInputSize((FRAME_WIDTH, FRAME_HEIGHT)) 
        

        success, faces = face_detector.detect(frame)

        target_pan_angle = PAN_CENTER
        target_tilt_angle = TILT_CENTER
        emotion_text = "Searching..."
        emotion_color = (255, 255, 255)

        if faces is not None:
            (x, y, w, h) = map(int, faces[0][:4]) 
            
            
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            pan_offset = pan_pid(face_center_x)
            tilt_offset = tilt_pid(face_center_y)
            
            
            target_pan_angle = PAN_CENTER + pan_offset  
            target_tilt_angle = TILT_CENTER - tilt_offset 
            
            
            x_end = min(x + w, FRAME_WIDTH)
            y_end = min(y + h, FRAME_HEIGHT)
            x_start = max(0, x)
            y_start = max(0, y)

            roi_color = frame[y_start:y_end, x_start:x_end]
            
            if roi_color.size > 0:
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(roi_gray, (TFLITE_INPUT_W, TFLITE_INPUT_H), interpolation=cv2.INTER_AREA)
                input_data = resized_face.astype('float32') / 255.0
                input_data = np.expand_dims(input_data, axis=0) 
                input_data = np.expand_dims(input_data, axis=-1)
                
                if input_data.shape != tuple(INPUT_SHAPE):
                    input_data = input_data.reshape(INPUT_SHAPE) 

            
                interpreter.set_tensor(input_details[0]['index'], input_data) 
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                max_index = np.argmax(predictions[0])
                max_confidence = predictions[0][max_index] 
                
                if max_confidence >= CONFIDENCE_THRESHOLD:
                    predicted_emotion = EMOTION_LABELS[max_index]
                    emotion_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
                    emotion_color = (0, 255, 0)
                else:
                    emotion_text = "Uncertain"
                    emotion_color = (255, 255, 255) 


           
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

        current_pan_angle = (target_pan_angle * SMOOTHING_FACTOR) + (current_pan_angle * (1.0 - SMOOTHING_FACTOR))
        current_tilt_angle = (target_tilt_angle * SMOOTHING_FACTOR) + (current_tilt_angle * (1.0 - SMOOTHING_FACTOR))
        
    
        current_pan_angle = max(0, min(180, current_pan_angle))
        current_tilt_angle = max(0, min(180, current_tilt_angle))

       
        pan_servo.angle = current_pan_angle
        tilt_servo.angle = current_tilt_angle

        
        cv2.circle(frame, (int(PAN_SETPOINT), int(TILT_SETPOINT)), 5, (0, 0, 255), -1) # Red dot at frame center
        
        
        cv2.imshow('YuNet Face Tracking & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    cleanup_and_exit()