from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
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

import argparse

# exemple :
# python3 save_keras_checkpoint_to_weights.py --weight_path "../ssd_keras_files/ssd300_OID_plates_2105_epoch-48_loss-9.1056_val_loss-8.2937.h5" --dest_path "../ssd_keras_files/ssd300_OID_plates_2105_MODEL.h5" --nclasse 1
parser = argparse.ArgumentParser(description='Save keras checkpoint .h5 file (190Mo) to weight model .h5 (80Mo) for inference')
parser.add_argument("--nclasse", help="The number of positive classes in the model", type=int, required=True)
parser.add_argument("--weight_path", help="Path to the input file to be converted", required=True)
parser.add_argument("--dest_path", help="Path to the model destination file", required=True)
parser.add_argument("--confidence", help="The confidence threshold for Non-maximum suppresion", required=False, type=float, default=0.2)
parser.add_argument("--iou", help="The IoU threshold", required=False, type=float, default=0.45)

args = parser.parse_args()

weight_path_input = args.weight_path
model_path_output = args.dest_path
confidence_thresh=args.confidence
iou_threshold=args.iou
n_classes = args.nclasse


# Set the image size.
img_height = 300
img_width = 300

scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.

# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

# 2: Load the trained weights into the model.

model.load_weights(weight_path_input, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

import json
from keras.models import load_model

"""
config = model.get_config()
print(config)
with open('../ssd_keras_files/model.config', 'w+') as f:
    json.dump(config, f)

json_config = model.to_json()
with open('../ssd_keras_files/model.json', 'w') as json_file:
    json_file.write(json_config)
"""

model.save(filepath=model_path_output)