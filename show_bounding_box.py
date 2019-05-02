from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_patch_sampling_ops import RandomMaxCropFixedAR
from data_generator.object_detection_2d_geometric_ops import *
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 3 # Number of color channels of the input images

dataset = DataGenerator()

# TODO: Set the paths to your dataset here.
images_path = '../../Datasets/OpenImages_face_plate/train/Vehicle registration plate/'
labels_path = '../../Datasets/OpenImages_face_plate/train/Vehicle registration plate/To_PASCAL_XML/'
image_set_filename = '../../Datasets/OpenImages_face_plate/train/Vehicle registration plate/ImageSets/filenames_xml.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'Human face',
           'Vehicle registration plate']

dataset.parse_xml(images_dirs=[images_path],
                  image_set_filenames=[image_set_filename],
                  annotations_dirs=[labels_path],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

print("Number of images in the dataset:", dataset.get_dataset_size())

convert_to_3_channels = ConvertTo3Channels()
random_max_crop = RandomMaxCropFixedAR(patch_aspect_ratio=img_width/img_height)
resize = Resize(height=img_height, width=img_width)

ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width)
random_flip           = RandomFlip(dim='vertical', prob=0.5)
random_translate      = RandomTranslate()
random_scale          = RandomScale()
random_rotate         = RandomRotate()


data_augmentation = [
                    ssd_data_augmentation
                    ]

generator = dataset.generate(batch_size=1,
                             shuffle=False, #image random in the dataset
                             transformations=data_augmentation,
                             returns={'processed_images',
                                      'processed_labels',
                                      'filenames'},
                             keep_images_without_gt=False)
                             

# Generate samples

batch_images, batch_labels, batch_filenames = next(generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])

# Visualize the boxes.

#%matplotlib inline

plt.figure(figsize=(20,12))
plt.imshow(batch_images[i])

current_axis = plt.gca()

# Draw the ground truth boxes in green (omit the label for more clarity)
for box in batch_labels[i]:
    class_id = box[0]
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(class_id)])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    #current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

plt.show()
