"""Implementation of useful summary ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re
from math import sqrt
from util import tensor_name


def summarize_gradients(grads_and_vars, name=None):
    """Adds histogram summaries for input list.

    Args:
      grads_and_vars: List, a list of tuples in form (grad, var).e
      name: String, name to use for var scope.
    """
    with tf.name_scope(name, 'gradients', grads_and_vars):
        for g, v in grads_and_vars:
            tf.summary.histogram(v.name + '/gradient', g)


def activation_summary(x, rows=0, cols=0, montage=True, name=None):
    """Summarize activations of input tensor.

    Args:
      x: Tensor, the input tensor to summarize. 
      rows: Integer, number of rows in montage (if applicable).
      cols: Integer, number of columns in montage (if applicable).
      montage: Boolean, whether to generate an image montage of this tensor.
    """
    n = tensor_name(x)
    # n = re.sub('tower_[0-9]*/', '', x.op.name).split('/')[-1]
    with tf.name_scope(name, 'activations', [x]):
        tf.summary.histogram(n + '/activations/histogram', x)
        tf.summary.scalar(n + '/activations/sparsity', tf.nn.zero_fraction(x))
        if montage:
            montage_summary(tf.transpose(x[0], [2, 0, 1]), name=n + '/activations')
            

def factorization(n):
    """Finds factors of n suitable for image montage.

    Args:
      n: Integer, value to factorize. Should not be prime!

    Returns: 
      A tuple of form (m, n) representing the number of rows and
      columns this number factorizes to.
    """
    for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0: #and i > 1:
            return (i, int(n/i))
    # raise ValueError("Invalid montage grid size of {}".format(n))
        
        
def montage_summary(x, m=0, n=0, name=None):
    """Generates a m x n image montage from the given tensor.

    If m or n is 0, attempts to infer the input shape at run time.  Note
    that this may not always be possible.

    Args:
      x: Tensor, images to combine in montage. 
      m: Integer, number of rows in montage grid.
      n: Integer, number of columns in montage grid.
      name: String, name of this summary.

    Returns:
      Summary image node containing the montage.
    """
    if n == 0 or m == 0:
        m, n = factorization(x.get_shape()[0].value)

    with tf.name_scope(name, 'montage', [x]):
        images = tf.split(x, n, axis=0)
        images = tf.concat(images, axis=1)
        images = tf.unstack(images, m, axis=0)
        images = tf.concat(images, axis=1)
        # if this is a b/w images, add a third dimension
        if len(images.shape) < 3:
            images = tf.expand_dims(images, axis=2)
        # add a first dimension for expected TB format
        images = tf.expand_dims(images, axis=0)
        y = tf.summary.image(name, images)
    return y

