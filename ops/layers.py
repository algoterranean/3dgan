"""Implementation of common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import batch_norm
import re
from util.scoping import tensor_name


def _weight_name(name):
    """Returns variable name with default suffix for weights."""
    return name if name is None else name + '/weights'


def _bias_name(name):
    """Returns variable name with default suffix for biases."""
    return name if name is None else name + '/bias'


@add_arg_scope
def dense(x,
          input_size,
          output_size,
          init=xavier_initializer,
          use_batch_norm=False,
          activation=None,
          reuse=False,
          name=None):
    """Standard dense (fully connected) layer. 

    Args:
      x: Tensor, the input tensor.
      input_size: Integer, number of input neurons.
      output_size: Integer, number of output neurons.
      init:
      use_batch_norm: Boolean, whether to use batch normalizationn.
      activation: Operation, activation function.
      reuse: Boolean, whether to reuse variables.
      name: String, name of output tensor.

    Returns:
      Tensor representing output of fully connected layer.
    """
    with tf.name_scope(name, 'dense', [x]):
        w_name, b_name = _weight_name(name), _bias_name(name)
        with tf.variable_scope('vars', reuse=reuse):
            w = tf.get_variable(name=w_name,
                                shape=[input_size, output_size],
                                initializer=init())
            b = tf.get_variable(name=b_name,
                                shape=[output_size],
                                initializer=init())
            if not reuse:
                tf.add_to_collection('weights', w)
                tf.add_to_collection('biases', b)
        h = tf.matmul(x, w) + b
        h = batch_norm(h) if use_batch_norm else h
        h = activation(h) if activation else h
        if not reuse:
            tf.add_to_collection('dense_layers', h)
    return h


@add_arg_scope
def conv2d(x,
           input_size,
           output_size,
           filter_size=3,
           stride=1,
           init=xavier_initializer,
           use_batch_norm=False,
           activation=None,
           reuse=False,
           name=None):
    """Standard 2D convolutional layer.

    Args:
      x: Tensor, input tensor.
      input_size: Integer, number of filters in input tensor.
      output_size: Integer, number of filters in output tensor.
      filter_size: Integer, size of convolution filter.
      stride: Integer, width/height convolution striding. Must be at least 1.
      init:
      use_batch_norm: Boolean, whether to use batch normalizationn.
      activation: Operation, activation function.
      reuse: Boolean, whether to reuse variables.
      name: String, name of output tensor.

    Returns:
      Tensor representing the output of the convolution layer.
    """
    with tf.name_scope(name, 'conv2d', [x]):
        w_name, b_name = _weight_name(name), _bias_name(name)
        with tf.variable_scope('vars', reuse=reuse):
            k = tf.get_variable(name=w_name,
                                shape=[filter_size, filter_size, input_size, output_size],
                                initializer=init())
            b = tf.get_variable(name=b_name,
                                shape=[output_size],
                                initializer=init())
            if not reuse:
                tf.add_to_collection('weights', k)
                tf.add_to_collection('biases', b)            
        h = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding='SAME')
        h = tf.nn.bias_add(h, b)
        h = batch_norm(h) if use_batch_norm else h
        h = activation(h) if activation else h
        if not reuse:
            tf.add_to_collection('conv_layers', h)
    return h


@add_arg_scope
def deconv2d(x,
             input_size,
             output_size,
             filter_size=3,
             stride=2,
             init=xavier_initializer,
             use_batch_norm=False,
             activation=None,
             reuse=False,
             name=None):
    """Standard fractionally strided (deconvolutional) layer.

    Args:
      x:
      input_size:
      output_size:
      filter_size:
      stride:
      init:
      use_batch_norm: Boolean, whether to use batch normalizationn.
      activation: Operation, activation function.
      reuse: Boolean, whether to reuse variables.
      name: String, name of output tensor.    

    Returns:
      Tensor representing the output of the deconvolution layer.
    """
    with tf.name_scope(name, 'deconv2d', [x]):
        w_name, b_name = _weight_name(name), _bias_name(name)
        with tf.variable_scope('vars', reuse=reuse):
            k = tf.get_variable(w_name,
                                [filter_size, filter_size, output_size, input_size],
                                initializer=init())
            b = tf.get_variable(b_name,
                                [output_size],
                                initializer=init())
            if not reuse:
                tf.add_to_collection('weights', k)
                tf.add_to_collection('biases', b)
        input_shape = tf.shape(x)
        output_shape = tf.stack([input_shape[0], input_shape[1]*2, input_shape[2]*2, output_size])
        h = tf.nn.conv2d_transpose(x,
                                   k,
                                   output_shape=output_shape,
                                   strides=[1, stride, stride, 1],
                                   padding='SAME')
        h = tf.nn.bias_add(h, b)
        h = batch_norm(h) if use_batch_norm else h
        h = activation(h) if activation else h
        if not reuse:
            tf.add_to_collection('conv_layers', h)
    return h


@add_arg_scope
def flatten(x,
            name=None):
    """Flattens input tensor while preserving batch size.

    Args:
      x: Tensor, the input tensor to flatten
      name: String, name of this operation.

    Returns:
      A flattened tensor that preserves the batch size.
    """
    with tf.name_scope(name, 'flatten', [x]):
        output_size = tf.reduce_prod(tf.shape(x)[1:])
        y = tf.reshape(x, [-1, output_size])
    return y
