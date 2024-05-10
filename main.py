from drowsy_function import drowsy
# from distracted_function import distracted
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        while counter < 200:
            null, frame = cap.read()
            # True when the person is sleepy
            drowsy_bool = drowsy(frame=frame)
            distracted_bool = False
            if drowsy_bool or distracted_bool:
                counter += 1
            elif counter > 0:
                counter -= 1
        
        counter = 0

        
