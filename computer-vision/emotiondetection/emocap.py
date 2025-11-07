import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

MODEL_PATH = 'media.tflite'
EMOTION_LABELS = ['Happy', 'Smile']
CONFIDENCE_THRESHOLD = 0.50 


try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'] 


HAAR_CASCADE_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
if face_cascade.empty():
    print(f"Error loading Haar Cascade file at: {HAAR_CASCADE_FILE}")
    exit()

print("Setting up Picamera2...")
try:
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(main={"size": (640, 480)}) 
    picam2.configure(config)
    
    picam2.start()
except Exception as e:
    print(f"Error initializing Picamera2: {e}")
    exit()

print("Camera started. Press 'q' to exit.")


while True:
    frame = picam2.capture_array() 

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
       
        resized_face = cv2.resize(roi_gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
        input_data = resized_face.astype('float32') / 255.0
        
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = input_data.reshape(INPUT_SHAPE) 
        
        
        interpreter.set_tensor(input_details[0]['index'], input_data) 
        interpreter.invoke() 

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        max_index = np.argmax(predictions)
        max_confidence = predictions[max_index]
        
        
        if max_confidence >= CONFIDENCE_THRESHOLD:
            predicted_emotion = EMOTION_LABELS[max_index]
            confidence_text = f"{predicted_emotion}: {max_confidence*100:.1f}%"
        else:
            confidence_text = f"Uncertain: {max_confidence*100:.1f}%"

        print(confidence_text)

    print("nofaces detected")

picam2.stop() 
cv2.destroyAllWindows()
print("Program terminated.")