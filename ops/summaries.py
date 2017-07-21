"""Implementation of useful summary ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re
from math import sqrt
from util.scoping import tensor_name


def summarize_activations():
    with tf.variable_scope('activations'):
        for l in tf.get_collection('conv_layers'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))
            montage_summary(tf.transpose(l[0], [2, 0, 1]), name=tensor_name(l) + '/montage')
        for l in tf.get_collection('dense_layers'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))

            
def summarize_losses():
    with tf.variable_scope('loss'):
        for l in tf.get_collection('losses'):
            tf.summary.scalar(tensor_name(l), l)
            tf.summary.histogram(tensor_name(l), l)

            
def summarize_weights_biases():
    with tf.variable_scope('weights'):
        for l in tf.get_collection('weights'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))
            # montage_summary(l, name=tensor_name(l) + '/montage')
    with tf.variable_scope('biases'):
        for l in tf.get_collection('biases'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))    
    

def summarize_gradients(grads_and_vars, name=None):
    """Adds histogram summaries for input list.

    Args:
      grads_and_vars: List, a list of tuples in form (grad, var).e
      name: String, name to use for var scope.
    """
    with tf.name_scope(name, 'gradients', grads_and_vars):
        for g, v in grads_and_vars:
            tf.summary.histogram(tensor_name(v) + '/gradient', g)

def summarize_collection(name, scope):
    """Add a scalar summary for every tensor in a collection."""
    collection = tf.get_collection(name, scope)
    for x in collection:
        tf.summary.scalar(tensor_name(x), x)
    return collection



# def activation_summary(x, rows=0, cols=0, montage=True, name=None):
#     """Summarize activations of input tensor.

#     Args:
#       x: Tensor, the input tensor to summarize. 
#       rows: Integer, number of rows in montage (if applicable).
#       cols: Integer, number of columns in montage (if applicable).
#       montage: Boolean, whether to generate an image montage of this tensor.
#     """
#     n = tensor_name(x)
#     # n = re.sub('tower_[0-9]*/', '', x.op.name).split('/')[-1]
#     # with tf.name_scope(name, 'activations', [x]) as scope:
        
#     tf.summary.histogram('activations', x)
#     tf.summary.scalar('activations/sparsity', tf.nn.zero_fraction(x))
#     if montage:
#         montage_summary(tf.transpose(x[0], [2, 0, 1]), 'montage')
        
#         # tf.summary.histogram(n + '/activations/histogram', x)
#         # tf.summary.scalar(n + '/activations/sparsity', tf.nn.zero_fraction(x))
#         # if montage:
#         #     montage_summary(tf.transpose(x[0], [2, 0, 1]), name=n + '/activations')
            

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
            return i, int(n/i)
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

    with tf.name_scope(name, 'montage', [x]) as scope:
        images = tf.split(x, n, axis=0)
        images = tf.concat(images, axis=1)
        images = tf.unstack(images, m, axis=0)
        images = tf.concat(images, axis=1)
        # if this is a b/w images, add a third dimension
        if len(images.shape) < 3:
            images = tf.expand_dims(images, axis=2)
        # add a first dimension for expected TB format
        images = tf.expand_dims(images, axis=0)
        y = tf.summary.image(scope, images)
    return y

