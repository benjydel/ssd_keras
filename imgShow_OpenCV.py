# Needed libraries
import argparse

import numpy as np
import random
import cv2

from threading import Thread
from datetime import datetime
import time


# exemple :
# python3 imgShow_OpenCV.py --file "../ssd_keras_files/Test AR DOD RC500S A6.mp4"
parser = argparse.ArgumentParser(description='Run OpenCV on i.MX6')
parser.add_argument("--file", help="Path to the file (Image or Video)", required=True)

args = parser.parse_args()

file_path = args.file
file_is_image = (cv2.imread(file_path) is not None)
print(file_is_image)


class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
          # if the thread indicator variable is set, stop the thread
          if self.stopped:
            return
    
          # otherwise, read the next frame from the stream
          (self.grabbed, self.frame) = self.stream.read()
    
    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        self.stopped = True

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec), (0, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


start = time.time()
cap = cv2.VideoCapture(file_path)
####TIME####
timebreak = time.time()
seconds = timebreak - start
print("Time VideoCapture: {0} seconds".format(seconds))
############
cps = CountsPerSec().start()

while True:
    start = time.time()
    (grabbed, frame) = cap.read()
    ####TIME####
    timebreak = time.time()
    seconds = timebreak - start
    print("Time cap.read: {0} seconds".format(seconds))
    ############
    if not grabbed or cv2.waitKey(1) == ord("q"):
        break
    
    if file_is_image is not True :
        frame = putIterationsPerSec(frame, cps.countsPerSec())

    start = time.time()
    cv2.imshow("Inference on "+file_path, frame)
    ####TIME####
    timebreak = time.time()
    seconds = timebreak - start
    print("Time imshow: {0} seconds".format(seconds))
    ############
    cps.increment()

if file_is_image is True :
    cv2.waitKey()
    
cv2.destroyAllWindows()