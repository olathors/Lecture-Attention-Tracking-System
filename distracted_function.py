import cv2
from gaze_tracking import GazeTracking

def distracted(frame, gaze):

    gaze.refresh(frame)


    if gaze.is_right():
        text = "Looking RIGHT"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        return True        
    elif gaze.is_left():
        text = "Looking LEFT"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        return True 
    elif gaze.is_up():
        text = "Looking ABOVE"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        return True
    elif gaze.is_down():
        text = "Looking BELOW"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        return True
    else:
        text = "Paying attention"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        return False
    
    