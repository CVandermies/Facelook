import numpy as np
import cv2
from math import degrees


def computeDistance(origin, destination):
    """Find the distance between two points in the Cartesian coordinates"""
    distance = origin - destination
    distance = np.sqrt(np.sum(np.multiply(distance, distance)))
    return distance


def align(img):
    """Find the rotation angle based on eyes position"""
    eye_finder = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    detected_face_toGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_finder.detectMultiScale(detected_face_toGray, 1.1, 10)
    
    # If less than 2 eyes detected -> None 
    if len(eyes) >= 2:
        eye_1 = eyes[0]; eye_2 = eyes[1]

        # Detect which eye is Left or Right based on position x
        if eye_1[0] < eye_2[0]:
            L_eye = eye_1; R_eye = eye_2
        else:
            L_eye = eye_2; R_eye = eye_1

        # Find midpoint of eyes as (x + w/2) and (y + h/2)
        L_eye = (int(L_eye[0] + (L_eye[2] / 2)), int(L_eye[1] + (L_eye[3] / 2)))
        R_eye = (int(R_eye[0] + (R_eye[2] / 2)), int(R_eye[1] + (R_eye[3] / 2)))

        # Find rotation direction
        L_eye_x, L_eye_y = L_eye
        R_eye_x, R_eye_y = R_eye
        
        if L_eye_y > R_eye_y:
            triangle_point = (R_eye_x, L_eye_y)
            direction = -1 # Rotate clockwise (antitrigo)
        else:
            triangle_point = (L_eye_x, R_eye_y)
            direction = 1 # Rotate counter-clockwise 

        a = computeDistance(np.array(L_eye), np.array(triangle_point))
        b = computeDistance(np.array(R_eye), np.array(triangle_point))
        c = computeDistance(np.array(R_eye), np.array(L_eye)) # Hypotenuse
        
        # If head is not horizontal, apply cosine rule to compute angle
        if b != 0 and a != 0: 
            cos_theta = (b*b + c*c - a*a)/(2*b*c)
            angle = degrees(np.arccos(cos_theta))
            if direction == -1:
                angle = 90 - angle
            return direction * angle
        return None
    return None 

    
def resize(image, target_size=(48,48), to_gray=True):	
    """Transform the matrix size to reach the target size"""
    image = cv2.resize(image, target_size)
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis = 0)
        image = np.divide(image, 255) 
    return image


def findFaces(frame, fcascade):
    """Determine the roi (region of interest) in the frame and process it"""
    result = []
    faces = []
    # List of list containing 4 points around the face
    faces = fcascade.detectMultiScale(frame, 1.1, 3)

    if len(faces) > 0:
        for x,y,w,h in faces:
            buff = frame[int(y):int(y+h), int(x):int(x+w)]            
            rotation = align(buff)
            resized_face = resize(buff)
            result.append((resized_face, [x, y, w, h], rotation))
        return result
    return None
