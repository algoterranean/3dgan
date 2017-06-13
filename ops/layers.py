"""Implementation of common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope
from tensorflow.contrib.layers import xavier_initializer as xav_init
from tensorflow.contrib.layers import variance_scaling_initializer as he_init
from tensorflow.contrib.layers import batch_norm
import re
from ops.summaries import activation_summary
from util import tensor_name



def weight_name(name):
    return name if name is None else name + '_w'


def bias_name(name):
    return name if name is None else name + '_b'


@add_arg_scope
def dense(x,
          input_size,
          output_size,
          use_batch_norm=False,
          activation=None,
          add_summaries=False,
          reuse=False,
          name=None):
    """Standard dense (fully connected) layer. """
    with tf.name_scope(name, 'dense', [x]):
        w_name, b_name = weight_name(name), bias_name(name)
        with tf.variable_scope('vars', reuse=reuse):
            W = tf.get_variable(name=w_name, shape=[input_size, output_size], initializer=xav_init())
            b = tf.get_variable(name=b_name, shape=[output_size], initializer=xav_init())
            if add_summaries:
                tf.summary.histogram(w_name, W)
                tf.summary.histogram(b_name, b)
            h = tf.matmul(x, W) + b
        h = batch_norm(h) if use_batch_norm else h
        h = activation(h) if activation else h
    return h


@add_arg_scope
def conv2d(x,
           input_size,
           output_size,
           ksize=3,
           stride=1,
           use_batch_norm=False,
           activation=None,
           add_summaries=False,
           reuse=False,
           name=None):
    """Standard 2D convolutional layer."""
    with tf.name_scope(name, 'conv2d', [x]):
        w_name, b_name = weight_name(name), bias_name(name)
        with tf.variable_scope('vars', reuse=reuse):
            K = tf.get_variable(name=w_name, shape=[ksize, ksize, input_size, output_size], initializer=xav_init())
            b = tf.get_variable(name=b_name, shape=[output_size], initializer=xav_init())
            if add_summaries:
                tf.summary.histogram(w_name, K)
                tf.summary.histogram(b_name, b)
        h = tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME')
        h = tf.nn.bias_add(h, b)
        h = batch_norm(h) if use_batch_norm else h
        h = activation(h) if activation else h
        if add_summaries:
            activation_summary(h)
    return h


@add_arg_scope
def deconv2d(x,
             input_size,
             output_size,
             ksize=3,
             stride=2,
             use_batch_norm=False,
             activation=None,
             add_summaries=False,
             reuse=False,
             name=None):
    """Standard fractionally strided (deconvolutional) layer."""
    with tf.name_scope(name, 'deconv2d', [x]):
        w_name, b_name = weight_name(name), bias_name(name)
        with tf.variable_scope('vars', reuse=reuse):
            K = tf.get_variable(w_name, [ksize, ksize, output_size, input_size], initializer=xav_init())
            b = tf.get_variable(b_name, [output_size], initializer=xav_init())
            if add_summaries:
                tf.summary.histogram(w_name, K)
                tf.summary.histogram(b_name, b)
        input_shape = tf.shape(x)
        output_shape = tf.stack([input_shape[0], input_shape[1]*2, input_shape[2]*2, output_size])
        h = tf.nn.conv2d_transpose(x, K, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
        h = tf.nn.bias_add(h, b)
        h = batch_norm(h) if use_batch_norm else h
        h = activation(h) if activation else h
        if add_summaries:
            activation_summary(h)
    return h


@add_arg_scope
def flatten(x,
            name=None):
    """
    Flattens input tensor while preserving batch size.
    """
    with tf.name_scope(name, 'flatten', [x]):
        output_size = tf.reduce_prod(tf.shape(x)[1:])
        y = tf.reshape(x, [-1, output_size], name=name)
    return y
