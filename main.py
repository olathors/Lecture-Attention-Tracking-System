from drowsy_function import drowsy
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        null, frame = cap.read()
        drowsy_bool = drowsy(frame=frame)
        print(drowsy_bool)
