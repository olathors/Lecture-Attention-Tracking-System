import time
import subprocess

start = "vlc introduction_deep_learning_2023.mp4"
pause = "dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2   org.mpris.MediaPlayer2.Player.PlayPause"
play = "dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2   org.mpris.MediaPlayer2.Player.Play"
stop = "dbus-send --type=method_call --dest=org.mpris.MediaPlayer2.vlc /org/mpris/MediaPlayer2   org.mpris.MediaPlayer2.Player.Stop"

process = subprocess.Popen(start, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(5)
process = subprocess.Popen(pause, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(5)
process = subprocess.Popen(play, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(5)
process = subprocess.Popen(stop, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
