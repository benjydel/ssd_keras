from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
#from imageio import imread
import numpy as np
import cv2
from matplotlib import pyplot as plt

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

import time

# Set the image size.
img_height = 300
img_width = 300

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
#img_path = '/home/bende/Datasets/OpenImages_face_plate/validation/Vehicle registration plate/0c756c9366a8cb10.jpg'
img_path = '../ssd_keras_files/voiturefeu.jpg'
video_path = '../ssd_keras_files/Test AR DOD RC500S A6.mp4'
confidence_threshold = 0.1

# TODO: Set the path to the `.h5` file of the model to be loaded.
#model_path = '../ssd_keras_files/ssd300_OID_plates_epoch-111_loss-4.7148_val_loss-3.8296.h5'
model_path = '../ssd_keras_files/ssd300_OID_plates_epoch-97_loss-2.9400_val_loss-2.4888.h5'

def predict_on_image(read_image, model):
    start = time.time()

    img = image.img_to_array(cv2.resize(read_image, (img_height, img_width))) 

    y_pred = model.predict(np.array([img]))
    ####TIME####
    timebreak = time.time()
    seconds = timebreak - start
    print("Time taken to make predictions: {0} seconds".format(seconds))
    ############

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])

    return y_pred_thresh

# Display the image and draw the predicted boxes onto it.
def display_pedicted_image(read_image, y_pred_thresh):
    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background',
            'Vehicle plate registration']

    """
    plt.figure(figsize=(20,12))
    plt.imshow(orig_images[0])

    current_axis = plt.gca()

    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    plt.show()
    """
    # Visualize detected bounding boxes.
    for box in y_pred_thresh[0]:
        classId = int(box[0])
        score = box[1]
        color = colors[int(box[0])]
        xmin = int(box[2] * read_image.shape[1] / img_width)
        ymin = int(box[3] * read_image.shape[0] / img_height)
        xmax = int(box[4] * read_image.shape[1] / img_width)
        ymax = int(box[5] * read_image.shape[0] / img_height)
        cv2.blur(src=read_image[ymin:ymax, xmin:xmax],dst=read_image[ymin:ymax, xmin:xmax], ksize=(20,10))
        cv2.rectangle(read_image, (xmin, ymin), (xmax, ymax), color, thickness=2)
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.putText(read_image, label, (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1)

    cv2.imshow('SSD Keras Plates recognition', read_image)



# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=1,
                mode='inference_fast',
                l2_regularization=0.005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                        [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                        [1.0, 2.0, 0.5],
                                        [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=confidence_threshold,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

model.load_weights(model_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
"""
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
"""

"""
read_image = cv2.imread(img_path)
y_pred_thresh = predict_on_image(read_image, model)
display_pedicted_image(read_image, y_pred_thresh)
cv2.waitKey()
"""

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video_path)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#frames_to_skip = fps/5
frames_to_skip = 1
i=1
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    if i == frames_to_skip:
        y_pred_thresh = predict_on_image(frame, model)
        display_pedicted_image(frame, y_pred_thresh)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        i=0
    i+=1
     
  # Break the loop
  else: 
    break

 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
