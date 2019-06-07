import numpy as np
import tensorflow as tf

import random
import cv2
from datetime import datetime
import time

file_path = "../ssd_keras_files/Collection_VM.jpg"


confidence_threshold = 0.2

classes = ["Plates"]


#SSD300 PARAMETERS
img_height = 300
img_width = 300

n_classes = len(classes) # Number of positive classes
classes_n_background = ['background'] + classes
colors = [ (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(classes)) ] #Creation of random colors according to the positive class number


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

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec), (0, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

# Display the image and draw the predicted boxes onto it.
def display_pediction_image(frame, y_pred_thresh):
    # Visualize detected bounding boxes.
    for box in y_pred_thresh[0]:
        classId = int(box[0])-1
        score = box[1]
        color = colors[classId]
        xmin = abs(int(box[2] * frame.shape[1] / img_width))
        ymin = abs(int(box[3] * frame.shape[0] / img_height))
        xmax = abs(int(box[4] * frame.shape[1] / img_width))
        ymax = abs(int(box[5] * frame.shape[0] / img_height))
        cv2.blur(src=frame[ymin:ymax, xmin:xmax],dst=frame[ymin:ymax, xmin:xmax], ksize=(20,10))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
        label = '{}: {:.2f}'.format(classes[classId], box[1])
        cv2.rectangle(frame, (xmin, ymin-2), (int(xmin+len(label)*8.5), ymin-15), (255,255,255), thickness=cv2.FILLED)
        cv2.putText(frame, label, (xmin, ymin-2), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

   

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../ssd_keras_files/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
cap = cv2.VideoCapture(file_path)
cps = CountsPerSec().start()

while True:
    (grabbed, frame) = cap.read()
    if not grabbed or cv2.waitKey(1) == ord("q"):
        break
    
    
    frame = putIterationsPerSec(frame, cps.countsPerSec())


    input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.invoke()
    interpreter.set_tensor(input_details[0]['index'], frame)

    
    y_pred_thresh = interpreter.get_tensor(output_details[0]['index'])

    print(y_pred_thresh)
    #display_pediction_image(frame, y_pred_thresh)
    #cv2.imshow("Inference on "+file_path, frame)
    cps.increment()

#cv2.waitKey()
#cv2.destroyAllWindows()

