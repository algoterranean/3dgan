"""Implementation of useful image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import re
from math import sqrt
from util import tensor_name



def colorize(x, colormap=cv2.COLORMAP_JET):
    """Convert a grayscale image tensor to a color-mapped RGB image.

    Uses a Python wrapper around OpenCV, so don't use it except f
    or occassional summaries, as it's fairly slow (relatively).

    Suggested colormaps:

    cv2.COLORMAP_JET
    cv2.COLORMAP_HOT
    cv2.COLORMAP_COOL
    cv2.COLORMAP_RAINBOW
    """    
    x = tf.cast(x * 255.0, tf.uint8)

    def helper(x):
        color_images = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3), dtype='uint8')
        for i in range(x.shape[0]):
            color_images[i] = cv2.applyColorMap(x[i], colormap)
        return color_images
    
    x = tf.py_func(helper, [x], tf.uint8)
    x = tf.reshape(x, [-1, 64, 64, 3])
    return x





