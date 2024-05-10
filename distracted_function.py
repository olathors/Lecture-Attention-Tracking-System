import cv2
from gaze_tracking import GazeTracking

def distracted(frame, gaze):

    gaze.refresh(frame)

    if gaze.is_right():
        return True        
    elif gaze.is_left():
        return True 
    else:
        return False