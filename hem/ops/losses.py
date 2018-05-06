"""Implementation of common loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rmse(x, x_hat, name='rmse'):
    return tf.sqrt(tf.reduce_mean(tf.square(x_hat - x)), name=name)


def rmse_scale_invariant(x, x_hat, name='rmse_scale_invariant'):
    return tf.multiply(0.5, (rmse(x, x_hat) + tf.reduce_mean(x_hat - x)), name=name)
