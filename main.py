from drowsy_function import drowsy
from distracted_function import distracted
import cv2
from gaze_tracking import GazeTracking
import time
import subprocess
commands = {
    "start":"cvlc introduction_deep_learning_2023.mp4",
    "pause":"dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2   org.mpris.MediaPlayer2.Player.PlayPause",
    "play":"dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2   org.mpris.MediaPlayer2.Player.Play",
    "stop":"dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2   org.mpris.MediaPlayer2.Player.Stop"
}


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    lecture_file = "introduction_deep_learning_2023.mp4"
    gaze = GazeTracking()
    counter = 0
    sec_per_loop = 0.1
    process = subprocess.Popen(commands["start"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        while counter < 50:
            # 1. Setup the timing: https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/
            # 2. Start the video player
            # 3. Give a notification about how close the user is to having to close the video
            start_time = time.time()
            null, frame = cap.read()
            # True when the person is sleepy
            drowsy_bool = drowsy(frame=frame)
            # distracted_bool = distracted(frame=frame, gaze=gaze)
            distracted_bool = False
            if drowsy_bool or distracted_bool:
                counter += 1
            elif counter > 0:
                counter -= 1
            end_time = time.time()
            elapsed_time = start_time - end_time
            remaining_time = sec_per_loop - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

        process = subprocess.Popen(commands["pause"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # pauses the video
        # starts the code for the handpose estimator 
        time.sleep(5)
        counter = 0
        # starts the video again
        process = subprocess.Popen(commands["play"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        
