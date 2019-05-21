# Needed libraries
import argparse

import numpy as np
import random
import cv2

from threading import Thread
from datetime import datetime
import time



parser = argparse.ArgumentParser(description='Make inference on SSD 300 model with Keras, TensorFlow or OpenVINO')
parser.add_argument("--mode", help="tf for Tensorflow, keras or ov for OpenVINO", required=False, default="tf", choices=('tf', 'keras', 'ov'))
parser.add_argument("--model", help="The path to the model (.pf for tf, .h5 for keras, ...)", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--file", help="Path to the file (Image or Video)", required=True)
parser.add_argument("--confidence", help="The confidence threshold for predictions", required=False, type=int, default=0.5)
parser.add_argument("--display", help="Bool to either display or not the predictions", required=False, default=True, choices=(True, False))

args = parser.parse_args()
run_mode = args.mode
model_path = args.model
file_path = args.file
file_is_image = (cv2.imread(file_path) is not None)
print(file_is_image)
confidence_threshold = args.confidence
display_bool = args.display
classes = []
for class_name in args.classes.split(','):
    classes.append(class_name)

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

def predict_on_image_tf(frame, sess):
    start = time.time()

    img = cv2.resize(frame, (img_height, img_width))

    img = img.astype(float)
    
    #On affecte -1 en premiere dimension, ce qui correspond à un ? dans la premiere couche du model
    resized_shape_img = img.reshape([-1, 300, 300, 3])
    
    #To found the input and output layers :
    """
    print('\n===== ouptut operation names =====\n')
    for op in sess.graph.get_operations():
            print("Operation Name :",op.name)         # Operation name
            print("Tensor Stats :",str(op.values()))     # Tensor name
    """
    # inference by the model (op name must comes with :0 to specify the index of its input and output)
    tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')
    tensor_output = sess.graph.get_tensor_by_name('import/decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3:0')

    #Prediction :
    y_pred = sess.run(tensor_output, {tensor_input: resized_shape_img})

    ####TIME####
    timebreak = time.time()
    seconds = timebreak - start
    print("Time taken to make predictions: {0} seconds".format(seconds))
    ############

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    if y_pred_thresh[0].size != 0:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

    return y_pred_thresh

def predict_on_image(frame, model):
    start = time.time()

    img = image.img_to_array(cv2.resize(frame, (img_height, img_width))) 

    y_pred = model.predict(np.array([img]))
    ####TIME####
    timebreak = time.time()
    seconds = timebreak - start
    print("Time taken to make predictions: {0} seconds".format(seconds))
    ############

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    if y_pred_thresh:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

    return y_pred_thresh

# Display the image and draw the predicted boxes onto it.
def display_pediction_image(frame, y_pred_thresh):
    # Visualize detected bounding boxes.
    for box in y_pred_thresh[0]:
        classId = int(box[0])-1
        score = box[1]
        color = colors[classId]
        xmin = int(box[2] * frame.shape[1] / img_width)
        ymin = int(box[3] * frame.shape[0] / img_height)
        xmax = int(box[4] * frame.shape[1] / img_width)
        ymax = int(box[5] * frame.shape[0] / img_height)
        cv2.blur(src=frame[ymin:ymax, xmin:xmax],dst=frame[ymin:ymax, xmin:xmax], ksize=(20,10))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
        label = '{}: {:.2f}'.format(classes[classId], box[1])
        cv2.rectangle(frame, (xmin, ymin-2), (int(xmin+len(label)*8.5), ymin-15), (255,255,255), thickness=cv2.FILLED)
        cv2.putText(frame, label, (xmin, ymin-2), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

def run_on_file(file_path, model_or_sess) :
    cap = cv2.VideoCapture(file_path)
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break
        
        if file_is_image is not True :
            frame = putIterationsPerSec(frame, cps.countsPerSec())

        if(run_mode == "keras"):
            y_pred_thresh = predict_on_image(frame, model_or_sess)
        elif (run_mode == "tf"):
            y_pred_thresh = predict_on_image_tf(frame, model_or_sess)
        else :
            print("OpenVINO Inference")
        display_pediction_image(frame, y_pred_thresh)
        cv2.imshow("Inference on "+file_path, frame)
        cps.increment()

    cv2.waitKey()
    cv2.destroyAllWindows()

## IF KERAS
if(run_mode == "keras") :
    from keras import backend as K
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.optimizers import Adam
    from models.keras_ssd300 import ssd_300
    from keras_loss_function.keras_ssd_loss import SSDLoss
    from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
    from keras_layers.keras_layer_DecodeDetections import DecodeDetections
    from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
    from keras_layers.keras_layer_L2Normalization import L2Normalization
    from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
    from data_generator.object_detection_2d_data_generator import DataGenerator
    from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
    from data_generator.object_detection_2d_geometric_ops import Resize
    from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

    # 1: Build the Keras model
    K.clear_session() # Clear previous models from memory.
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                'L2Normalization': L2Normalization,
                                                'DecodeDetections': DecodeDetections,
                                                'compute_loss': ssd_loss.compute_loss})
    run_on_file(file_path, model)
        
# IF Tensorflow
elif(run_mode == "tf") :
    import tensorflow as tf
    from tensorflow.python.platform import gfile

    with tf.Session() as sess:
        # load model from pb file
        with gfile.FastGFile(model_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
        # write to tensorboard (check tensorboard for each op names)
        writer = tf.summary.FileWriter('../ssd_keras_files/log/')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()
        # print all operation names 

        run_on_file(file_path, sess)
# IF OPENVINO
else :
    print("OpenVINO")