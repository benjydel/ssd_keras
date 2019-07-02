import numpy as np
import time
import cv2
from threading import Thread
from datetime import datetime
import random

model_name = "../ssd_keras_files/frozen_inference_graph"
model_bin = "../ssd_keras_files/frozen_inference_graph.bin"
model_xml = "../ssd_keras_files/frozen_inference_graph.xml"
confidence_threshold = 0.5

file_path = "../ssd_keras_files/Test AR DOD RC500S A6.mp4"


file_is_image = (cv2.imread(file_path) is not None)
print(file_is_image)
confidence_threshold = 0.2
classes = ["plate"]

#SSD300 PARAMETERS
img_height = 300
img_width = 300

iou_threshold = 0.4
top_k = 200


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
    
    def reset(self):
        self._start_time = datetime.now()
        self._num_occurrences = 0

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

def predict_on_image_ov(frame, net):
    start = time.time()
    blob = cv2.dnn.blobFromImage(frame,1,(300,300))
    #print(blob.shape)
    print(blob.dtype)
    end = time.time()
    print("\t[INFO] blob took " + str((end-start)*1000) + " ms")
    
    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    start = time.time()
    net.setInput(blob)
    end = time.time()
    print("\t[INFO] set blob took " + str((end-start)*1000) + " ms")
    start = time.time()
    #pred : num_detections, detection_classes, detection_scores, detection_boxes (ymin, xmin, ymax, xmax)
    y_preds = net.forward()
    print(y_preds)
    end = time.time()
    print("\t[INFO] classification took " + str((end-start)*1000) + " ms")
    
    start = time.time()
    y_pred_arrange = np.squeeze(y_preds)[:,1:]
    #print(y_pred_arrange)
    
    y_pred_thresh = []
    for k in range(y_pred_arrange.shape[0]):
        if(y_pred_arrange[k][1] > confidence_threshold):
            y_pred_thresh.append(y_pred_arrange[k])

    y_pred_thresh = np.array(y_pred_thresh)
    end = time.time()
    print("\t[INFO] arrange array took " + str((end-start)*1000) + " ms")

    if y_pred_thresh.size != 0:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh)
    
    return y_pred_thresh

# Display the image and draw the predicted boxes onto it.
def display_pediction_image(frame, y_pred_thresh):
    # Visualize detected bounding boxes.
    for box in y_pred_thresh:
        classId = int(box[0])-1
        score = box[1]
        color = colors[classId]
        xmin = abs(int(box[2] * frame.shape[1]))
        ymin = abs(int(box[3] * frame.shape[0]))
        xmax = abs(int(box[4] * frame.shape[1]))
        ymax = abs(int(box[5] * frame.shape[0]))
        cv2.blur(src=frame[ymin:ymax, xmin:xmax],dst=frame[ymin:ymax, xmin:xmax], ksize=(20,10))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
        label = '{}: {:.2f}'.format(classes[classId], box[1])
        cv2.rectangle(frame, (xmin, ymin-2), (int(xmin+len(label)*8.5), ymin-15), (255,255,255), thickness=cv2.FILLED)
        cv2.putText(frame, label, (xmin, ymin-2), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

def run_on_file(file_path, model_or_sess) :
    cap = cv2.VideoCapture(file_path)
    cps = CountsPerSec().start()

    while True:
        start1 = time.time()
        (grabbed, frame) = cap.read()
        end = time.time()
        print("[INFO] Read image took " + str((end-start1)*1000) + " ms")
        start = time.time()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break
        end = time.time()
        print("[INFO] Wait key took " + str((end-start)*1000) + " ms")

        start = time.time()
        y_pred_thresh = predict_on_image_ov(frame, model_or_sess)
        end = time.time()
        print("[INFO] Prediction took " + str((end-start)*1000) + " ms")

        start = time.time()
        if file_is_image is not True :
            frame = putIterationsPerSec(frame, cps.countsPerSec())
        end = time.time()
        print("[INFO] put fps " + str((end-start)*1000) + " ms")

        start = time.time()
        display_pediction_image(frame, y_pred_thresh)
        end = time.time()
        print("[INFO] add bounding box took " + str((end-start)*1000) + " ms")
        start = time.time()
        cv2.imshow("Inference on "+file_path, frame)
        end = time.time()
        print("[INFO] display image took " + str((end-start)*1000) + " ms")
        cps.increment()

        end = time.time()
        print("[TOTAL]" + str((end-start1)*1000) + " ms\n")
        

    if file_is_image is True :
        cv2.waitKey()
    cv2.destroyAllWindows()



print("[INFO] loading model...")
net = cv2.dnn.readNet( model_name+".bin",model_name+".xml")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE) #
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

net.setInput(np.zeros((1,3,300,300), dtype = "float32"))
net.forward()

run_on_file(file_path, net)





