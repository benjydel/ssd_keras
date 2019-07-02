# Needed libraries
import argparse

from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

import numpy as np
import random
import cv2

from threading import Thread
from datetime import datetime
import time


# exemple :
# python3 inference_ssd300.py --model "../ssd_keras_files/ssd300_OID_plates_MODEL.h5" --classes "Vehicle registration plates" --file "../ssd_keras_files/Test AR DOD RC500S A6.mp4" --mode keras --confidence 0.2
# python3 inference_ssd300.py --model "../ssd_keras_files/ssd300_OID_plates_frozen_model.pb" --classes "plates" --file "../ssd_keras_files/Collection_VM.jpg" --mode tf
parser = argparse.ArgumentParser(description='Make inference on SSD 300 model with Keras, TensorFlow or OpenVINO')
parser.add_argument("--model", help="The path to the model (.pf for tf, .h5 for keras, ...)", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--file", help="Path to the file (Image or Video)", required=True)
parser.add_argument("--confidence", help="The confidence threshold for predictions", required=False, type=float, default=0.5)
parser.add_argument("--display", help="Bool to either display or not the predictions", required=False, default=True, choices=(True, False))

args = parser.parse_args()
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

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def predict_on_image_tf(frame, sess):
    start = time.time()

    img = cv2.resize(frame, (img_height, img_width))

    img = img.astype(float)
    
    #Add -1 on the first dimension which correspond to the interrogation mark "?" within the first model layer of frozen graph converted from keras
    resized_shape_img = img.reshape([-1, 300, 300, 3])
    
    #To found the input and output layers :
    
    i = 0
    print('\n===== ouptut operation names =====\n')
    for op in sess.graph.get_operations():
        print("Operation Name :",op.name)         # Operation name
        print("Tensor Stats :",str(op.values()))     # Tensor name
        i+=1
    
    # inference by the model (op name must comes with :0 to specify the index of its input and output)

    #tensor_input = sess.graph.get_tensor_by_name('input_1:0')
    #tensor_output = sess.graph.get_tensor_by_name('predictions/concat:0')
    tensor_input = sess.graph.get_tensor_by_name('image_tensor:0')
    tensor_output = sess.graph.get_tensor_by_name('BoxPredictor_0/BoxEncodingPredictor:0')

    #Prediction :
    y_pred = sess.run(tensor_output, {tensor_input: resized_shape_img})
    """
    y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=confidence_threshold,
                                   iou_threshold=iou_threshold,
                                   top_k=top_k,
                                   normalize_coords=True,
                                   img_height=img_height,
                                   img_width=img_width)
    """
    ####TIME####
    timebreak = time.time()
    seconds = timebreak - start
    print("Time taken to make predictions: {0} seconds".format(seconds))
    ############

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    if y_pred_thresh[0].count != 0:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

    return y_pred_thresh

# Display the image and draw the predicted boxes onto it.
def display_pediction_image(frame, y_pred_thresh):
    # Visualize detected bounding boxes.
    for box in y_pred_thresh:
        classId = int(box[0])-1
        score = box[1]
        color = colors[classId]
        xmin = abs(int(box[3] * frame.shape[1]))
        ymin = abs(int(box[2] * frame.shape[0]))
        xmax = abs(int(box[5] * frame.shape[1]))
        ymax = abs(int(box[4] * frame.shape[0]))
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
        
        

        #y_pred_thresh = predict_on_image_tf(frame, model_or_sess)
        
        image_np_expanded = np.expand_dims(frame, axis=0)
        output = run_inference_for_single_image(image_np_expanded, model_or_sess)
        y_pred = np.column_stack( 
                (np.column_stack(
                    (output['detection_classes'],
                    output['detection_scores'])
                ),
                output['detection_boxes'])
            )

        y_pred_thresh = []
        for k in range(y_pred.shape[0]):
            if(y_pred[k][1] > confidence_threshold):
                y_pred_thresh.append(y_pred[k])

        y_pred_thresh = np.array(y_pred_thresh)
        if y_pred_thresh.size != 0:
            np.set_printoptions(precision=2, suppress=True, linewidth=90)
            print("Predicted boxes:\n")
            print('   class   conf xmin   ymin   xmax   ymax')
            print(y_pred_thresh)

        if file_is_image is not True :
            frame = putIterationsPerSec(frame, cps.countsPerSec())

        display_pediction_image(frame, y_pred_thresh)
        cv2.imshow("Inference on "+file_path, frame)
        cps.increment()

    if file_is_image is True :
        cv2.waitKey()
    cv2.destroyAllWindows()




import tensorflow as tf
from tensorflow.python.platform import gfile
    

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_path, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

run_on_file(file_path, detection_graph)

