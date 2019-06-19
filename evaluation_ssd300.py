from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

########################################## TO CONFIGURE ##############################################
model_path = '../ssd_keras_files/4-ssd300_OID_plates_1000step_per_epoch_reduceOnPlateau.h5'
model_mode = 'training'
matching_iou_threshold = 0.5
batch_size = 8

images_dir = ['../../Datasets/OpenImages_face_plate/test/Vehicle registration plate/']
annotations_dir = ['../../Datasets/OpenImages_face_plate/test/Vehicle registration plate/To_PASCAL_XML/']
filename = ['../../Datasets/OpenImages_face_plate/test/Vehicle registration plate/ImageSets/filenames_xml.txt']

classes = ['Vehicle registration plate']

######################################################################################################

# Set a few configuration parameters.
img_height = 300
img_width = 300

n_classes = len(classes) # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
classes_n_background = ['background'] + classes

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})


dataset = DataGenerator()


dataset.parse_xml(images_dirs=images_dir,
                  image_set_filenames=filename,
                  annotations_dirs=annotations_dir,
                  classes=classes_n_background,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=batch_size,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=matching_iou_threshold,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes_n_background[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes: break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0,1,11))
        cells[i, j].set_yticks(np.linspace(0,1,11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(classes_n_background[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)

plt.show()