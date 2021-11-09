#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

# %%
# Download the test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
# First we will download the images that we will use throughout this tutorial. The code snippet
# shown bellow will download the test images from the `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
# and save them inside the ``data/images`` folder.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


PATH_TO_LABELS = './annotations/label_map.pbtxt'
# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# PATH_TO_SAVED_MODEL = "D:/AI/TrainingAlgo/Tesnsorflow_Object-Detection_Train_241/output_inference_graph/saved_model"
PATH_TO_SAVED_MODEL = "./exported-models/mjmetal/pretrained_model/resnet101_v1/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# %%
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

i=0
# image_np = load_image_into_numpy_array(image_path)
# image_np = cv2.imread("D:/AI/TF_Object_Detection_API/mjmetal/bmp_image_12.bmp")

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

PATH_TO_TEST_IMAGE = './images'
n_images = len(os.listdir(PATH_TO_TEST_IMAGE))
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGE, 'images_{}.jpg'.format(i+1)) for i in range(n_images)]

for image_path in TEST_IMAGE_PATHS:

      for_loop_time_start = time.perf_counter()
      image_np = load_image_into_numpy_array(image_path)
      # image_np = cv2.imread(image_path)

      input_tensor = tf.convert_to_tensor(image_np)
          # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis, ...]

          # input_tensor = np.expand_dims(image_np, 0)
      detections = detect_fn(input_tensor)

          # All outputs are batches tensors.
          # Convert to numpy arrays, and take index [0] to remove the batch dimension.
          # We're only interested in the first num_detections.
      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                         for key, value in detections.items()}
      detections['num_detections'] = num_detections

          # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

      image_np_with_detections = image_np.copy()

      for_loop_time_end = time.perf_counter()
      loop_time = (for_loop_time_end - for_loop_time_start)

      str99 = "For loop Time : %0.9f" % loop_time

      cv2.putText(image_np_with_detections, str99, (1, 32), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

      viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

      plt.figure()
      cv2.imshow("test_resnet101", image_np_with_detections)
      plt.imshow(image_np_with_detections)
      print('Done')

      if cv2.waitKey(0) == ord('q'):
            continue
      else:
            break

plt.show()


# sphinx_gallery_thumbnail_number = 2
