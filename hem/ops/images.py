"""Implementation of useful image ops."""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import tensorflow as tf


def colorize(x, colormap=cv2.COLORMAP_JET, name=None):
    """Convert a grayscale image tensor to a color-mapped RGB image.

    Uses a Python wrapper around OpenCV, so don't use it except
    for occasional summaries, as it's fairly slow (relatively).

    NOTE the output of this is in NHWC format.

    Suggested colormaps:
    cv2.COLORMAP_JET
    cv2.COLORMAP_HOT
    cv2.COLORMAP_COOL
    cv2.COLORMAP_RAINBOW

    Args:
        x: Tensor, input (grayscale) images
        colormap: Integer, a valid colormap from OpenCV
        name: String, name of the resulting Tensor

    Returns:
        Tensor, a colorized version of x
    """
    with tf.name_scope(name, 'colorize', [x]):
        x = tf.cast(x * 255.0, tf.uint8)
        # switch to NHWC mode
        x = tf.transpose(x, [0, 2, 3, 1])
        h, w = int(x.shape[1]), int(x.shape[2])
        if len(x.shape) <= 3:
            x = tf.expand_dims(x, axis=3)

        def helper(img):
            color_images = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3), dtype=img.dtype) #dtype='uint8')
            for i in range(img.shape[0]):
                color_images[i] = cv2.applyColorMap(img[i], colormap)
                # TODO set invalid values (max or min) as black
                # color_images[i][x[i] == 0] = (0, 0, 0)
            return color_images

        x = tf.py_func(helper, [x], tf.uint8)
        x = tf.reshape(x, [-1, h, w, 3])
    return x


def rescale(x, orig=(-1, 1), new=(0, 1), name=None):
    """
    Uniformly rescales tensor from original range to new range.

    Args:
        x: Tensor, the input tensor.
        orig: Tuple, the original range (min, max) of the tensor.
        new: Tuple, the new range (min, max) of the tensor.
        name: String, name of operation

    Returns:
        Tensor, the rescaled tensor.
    """
    with tf.name_scope(name, 'rescale', [x]):
        # print('input_shape', x.shape)
        result = (x - orig[0]) * (new[1] - new[0]) / (orig[1] - orig[0]) + new[0]
        # print('output_shape', result.shape)
    return result


def instance_norm(x, reuse=False, name=None):
    with tf.name_scope(name, 'instance_norm', [x]):
        b, c, h, w = [i.value for i in x.get_shape()]   # NCHW shape
        var_shape = [c]
        mu, sigma_sq = tf.nn.moments(x, [2, 3], keep_dims=True)
        with tf.variable_scope('vars', reuse=reuse):
            shift = tf.get_variable(name=name+'/shift', shape=var_shape, initializer=tf.zeros_initializer)
            scale = tf.get_variable(name=name+'/scale', shape=var_shape, initializer=tf.ones_initializer)
        # shift = tf.Variable(tf.zeros(var_shape))
        # scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (x-mu)/(sigma_sq + epsilon)**(.5)
    j = scale * tf.transpose(normalized, [0, 2, 3, 1])
    j = j + shift
    j = tf.transpose(j, [0, 3, 1, 2])
    return j
    # return scale * normalized + shift


def center_crop(x, fraction):
    x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.map_fn(lambda a: tf.image.central_crop(a, central_fraction=fraction), x)
    return tf.transpose(x, [0, 3, 1, 2])

def crop_to_bounding_box(x, x1, y1, w, h):
    x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.image.crop_to_bounding_box(x, x1, y1, w, h)
    x = tf.transpose(x, [0, 3, 1, 2])
    return x

