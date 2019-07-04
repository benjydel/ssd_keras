import numpy as np
import time
import cv2
from threading import Thread
from datetime import datetime
import random
import argparse

#exemple:
#python3 openvino_inference_ssd300.py --mode=tf_gpu --model_name=../ssd_keras_files/plate_inference_graph_retrained/frozen_inference_graph --classes="plates" --file=../ssd_keras_files/vehiculesutilitairesW.png --confidence=0.2
#python3 openvino_inference_ssd300.py --mode=ov --model_name=/home/root/ssd_keras_files/frozen_inference_graph --classes="plates" --file="/home/root/ssd_keras_files/Test AR DOD RC500S A6.mp4" --confidence=0.2
parser = argparse.ArgumentParser(description='Make inference on SSD 300 model TensorFlow or OpenVINO')
parser.add_argument("--mode", help="tf for Tensorflow, tf_gpu for tensorflow with GPU, or ov for OpenVINO", required=False, default="tf", choices=('tf', 'tf_gpu', 'ov'))
parser.add_argument("--model_name", help="The path to the model (Do not write the extension .pb, .bin, .xml ...)", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--file", help="Path to the file (Image or Video)", required=True)
parser.add_argument("--confidence", help="The confidence threshold for predictions", required=False, type=float, default=0.5)
parser.add_argument("--display", help="Bool to either display or not the predictions", required=False, default=True, choices=(True, False))

args = parser.parse_args()
run_mode = args.mode
model_name = args.model_name
file_path = args.file
file_is_image = (cv2.imread(file_path) is not None)
print(file_is_image)
confidence_threshold = args.confidence
display_bool = args.display
classes = []
for class_name in args.classes.split(','):
    classes.append(class_name)

n_classes = len(classes) # Number of positive classes
classes_n_background = ['background'] + classes
colors = [ (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(classes)) ] #Creation of random colors according to the positive class number


#SSD300 PARAMETERS
img_height = 300
img_width = 300

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

def run_inference_for_single_image(image, graph):
    start = time.time()
    #to set shape to [1, width, height, 3] instead of [width, height, 3]
    #tensorflow model already contain reshape function using while loop
    img_reshaped = np.expand_dims(image, axis=0)
    end = time.time()
    print("\t[INFO] Reshape took " + str((end-start)*1000) + " ms")

    start = time.time()
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            end = time.time()
            print("\t[INFO] set input and output " + str((end-start)*1000) + " ms")

            start = time.time()
            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: img_reshaped})
            end = time.time()
            print("\t[INFO] net forward took " + str((end-start)*1000) + " ms")

            start = time.time()
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
   
    y_pred = np.column_stack( 
            (np.column_stack(
                (output_dict['detection_classes'],
                output_dict['detection_scores'])
            ),
            output_dict['detection_boxes'])
        )
    
    end = time.time()
    print("\t[INFO] compute numpy array took " + str((end-start)*1000) + " ms")
    #change the order for xmin, xmax, ymin, ymax
    return y_pred[...,[0,1,3,2,5,4]]

def net_forward_cv2_openvino(frame, net):
    #to set shape to [1, width, height, 3] instead of [width, height, 3]
    #openVINO model does not contain reshape function because tensorflow uses while loop which OpenVINO does not support
    start = time.time()
    blob = cv2.dnn.blobFromImage(frame,1,(img_width,img_height))
    #print(blob.shape)
    print(blob.dtype)
    end = time.time()
    print("\t[INFO] Reshape took " + str((end-start)*1000) + " ms")
    
    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    start = time.time()
    net.setInput(blob)
    end = time.time()
    print("\t[INFO] set input " + str((end-start)*1000) + " ms")
    start = time.time()
    #pred : num_detections, detection_classes, detection_scores, detection_boxes (ymin, xmin, ymax, xmax)
    y_preds = net.forward()
    #print(y_preds)
    end = time.time()
    print("\t[INFO] net forward took " + str((end-start)*1000) + " ms")
    
    y_pred_arrange = np.squeeze(y_preds)[:,1:]

    return y_pred_arrange

def predict_on_image(frame, net_or_graph, run_mode):
    start = time.time()
    if(run_mode == "ov"):
        y_pred = net_forward_cv2_openvino(frame, net_or_graph)
    else:
        y_pred = run_inference_for_single_image(frame, net_or_graph)

    end = time.time()
    print("--> Total Prediction took " + str((end-start)*1000) + " ms")

    start = time.time()
    y_pred_thresh = []
    for k in range(y_pred.shape[0]):
        if(y_pred[k][1] > confidence_threshold):
            y_pred_thresh.append(y_pred[k])

    y_pred_thresh = np.array(y_pred_thresh)
    end = time.time()
    print("[INFO] keep threshold values array took " + str((end-start)*1000) + " ms")

    if y_pred_thresh.size != 0:
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   score xmin   ymin   xmax   ymax')
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

def run_on_file(file_path, net_or_graph, run_mode) :
    start = time.time()
    cap = cv2.VideoCapture(file_path,cv2.CAP_FFMPEG)
    end = time.time()
    print("[INFO] video capture took " + str((end-start)*1000) + " ms")
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

        y_pred_thresh = predict_on_image(frame, net_or_graph, run_mode)

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
## IF OpenVino
if(run_mode == "ov") :
    net = cv2.dnn.readNet( model_name+".bin",model_name+".xml")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE) #
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    #First pass of the network with an empty array to allocate all the memory space.
    net.setInput(np.zeros((1,3,300,300), dtype = "float32"))
    net.forward()
    net_or_graph = net
else :
    if(run_mode == "tf"):
        import os
        #disable de GPU visibility for tensorflow
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    import tensorflow as tf
    from tensorflow.python.platform import gfile
        
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_name+".pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    net_or_graph = detection_graph

run_on_file(file_path, net_or_graph, run_mode)





