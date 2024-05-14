from drowsy_function import drowsy
from distracted_function import distracted
import cv2
from gaze_tracking import GazeTracking
import time

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    lecture_file = "introduction_deep_learning_2023.mp4"
    gaze = GazeTracking()
    counter = 0
    sec_per_loop = 0.1
    while True:
        while counter < 50:
            # 1. Setup the timing: https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/
            # 2. Start the video player
            # 3. Give a notification about how close the user is to having to close the video
            start_time = time.time()
            null, frame = cap.read()
            # True when the person is sleepy
            drowsy_bool = drowsy(frame=frame)
            distracted_bool = distracted(frame=frame, gaze=gaze)
            if drowsy_bool or distracted_bool:
                counter += 1
            elif counter > 0:
                counter -= 1
            end_time = time.time()
            elapsed_time = start_time - end_time
            remaining_time = sec_per_loop - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

        print("Follow the video you lazy")        
        counter = 0

        
