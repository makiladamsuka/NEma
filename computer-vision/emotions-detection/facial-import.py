import os
import cv2
import numpy as np
from Face_Pose_Detection import get_face_landmarks
data_dir = './data'
output = []

for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image at {image_path}. Skipping.")
            continue 
        face_landmarks_data = get_face_landmarks(image)
        if face_landmarks_data is not None:
            EXPECTED_LENGTH = 1404
            if len(face_landmarks_data) == EXPECTED_LENGTH:
                face_landmarks_data.append(int(emotion_indx)) 
                output.append(face_landmarks_data)
                
            else:
                print(f"Skipping data: Found {len(face_landmarks_data)} landmarks, expected {EXPECTED_LENGTH}.")
        else:
            print(f"Skipping data: No face found in {image_path}.")
if output:
    np.savetxt('data.txt', np.asarray(output))
    print(f"Successfully saved {len(output)} records to data.txt")