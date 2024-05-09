import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
    return aspect_ratio_Eye

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_detector = dlib. get_fronatl_face_detector()

def drowsy(frame):
    """Checks if the the face in the frama is drowsy or not
    args:
        frame: the frame of the scene
    returns:
        drowsy (boolean): returns True for drowsy and False for awake
    """
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale,face)
        leftEye = []
        rightEye = []
        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))

        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_rat = round(Eye_rat,2)
        if Eye_rat < 0.25:
            return True
    return False