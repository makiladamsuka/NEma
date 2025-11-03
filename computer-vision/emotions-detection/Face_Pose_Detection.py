import cv2
import mediapipe as mp 

def get_face_landmarks(image, draw=False, static_image_mode=True):
    if image is None:
        return None
    try:
        image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return None 
        
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                max_num_faces=1,
                                                min_detection_confidence=0.5)
    results = face_mesh.process(image_input_rgb)
    image_landmarks = None 
    if results.multi_face_landmarks:
        image_landmarks = [] 
        ls_single_face = results.multi_face_landmarks[0].landmark
        xs = []
        ys = []
        zs = []
        for idx in ls_single_face:
            xs.append(idx.x)
            ys.append(idx.y)
            zs.append(idx.z)
        min_x = min(xs)
        min_y = min(ys)
        min_z = min(zs)
        for j in range(len(xs)):
            image_landmarks.append(xs[j] - min_x)
            image_landmarks.append(ys[j] - min_y)
            image_landmarks.append(zs[j] - min_z)
        if draw:
            pass
    return image_landmarks