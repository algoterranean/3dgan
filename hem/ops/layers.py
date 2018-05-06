"""Implementation of common layers."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import batch_norm as batch_norm_op
from hem.ops.images import instance_norm as instance_norm_op


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
          use_batch_renorm=False,
          activation=None,
          reuse=False,
          dropout=0,
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
      dropout:
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
        h = batch_norm_op(h, fused=True, data_format='NCHW', renorm=use_batch_renorm) if (use_batch_norm or use_batch_renorm) else h
        h = activation(h) if activation else h
        h = tf.nn.dropout(h, keep_prob=dropout) if (dropout > 0) else h
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
           use_batch_renorm=False,
           use_instance_norm=False,
           activation=None,
           reuse=False,
           dropout=0,
           padding='SAME',
           name=None):
    """Standard 2D convolutional layer.

    Args:
      x: Tensor, input tensor.
      input_size: Integer, number of filters in input tensor.
      output_size: Integer, number of filters in output tensor.
      filter_size: Integer, size of convolution filter.
      stride: Integer, width/height convolution striding. Must be at least 1.
      init: Function, initializer for weights and biases.
      use_batch_norm: Boolean, whether to use batch normalizationn.
      activation: Operation, activation function.
      reuse: Boolean, whether to reuse variables.
      dropout: Boolean, whether to use dropout.
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

        # h = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding='SAME', data_format='NCHW')
        strides = [1, 1, stride, stride]  # NCHW format
        h = tf.nn.conv2d(x, k, strides=strides, padding=padding, data_format='NCHW')
        h = tf.nn.bias_add(h, b, data_format='NCHW')
        if not reuse:
            tf.summary.scalar(name + '/conv_output', tf.reduce_mean(h))
            tf.summary.histogram(name + '/conv_output', h)
        h = instance_norm_op(h, reuse=reuse, name=name) if use_instance_norm else h
        h = batch_norm_op(h, fused=True, data_format='NCHW', renorm=use_batch_renorm) if (use_batch_norm or use_batch_renorm) else h
        if not reuse and use_batch_norm:
            tf.summary.scalar(name + '/batchnorm_output', tf.reduce_mean(h))
            tf.summary.histogram(name + '/batchnorm_output', h)
        h = activation(h) if activation else h
        if not reuse:
            tf.summary.scalar(name + '/activation', tf.reduce_mean(h))
            tf.summary.histogram(name + '/activation', h)
        h = tf.nn.dropout(h, keep_prob=dropout) if (dropout > 0) else h
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
             output_shape=None,
             use_batch_norm=False,
             use_batch_renorm=False,
             use_instance_norm=False,
             activation=None,
             reuse=False,
             dropout=0,
             padding='SAME',
             name=None):
    """Standard fractionally strided (deconvolutional) layer.

    Args:
      x:
      input_size: Integer, number of input channels.
      output_size: Integer, number of output channels.
      filter_size: Integer, convolution filter size.
      stride: Integer, convolution stride.
      init: Function, initializer for weights and biases.
      use_batch_norm: Boolean, whether to use batch normalizationn.
      activation: Operation, activation function.
      reuse: Boolean, whether to reuse variables.
      dropout: Boolean, whether to use dropout.
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

        if output_shape is None:
            input_shape = tf.shape(x)
            output_shape = tf.stack([input_shape[0], output_size, input_shape[2] * 2, input_shape[3] * 2])
        strides = [1, 1, stride, stride]
        h = tf.nn.conv2d_transpose(x,
                                   k,
                                   output_shape=output_shape,
                                   strides=strides,
                                   padding=padding,
                                   data_format='NCHW')
        h = tf.nn.bias_add(h, b, data_format='NCHW')
        if not reuse:
            tf.summary.scalar(name + '/deconv_output', tf.reduce_mean(h))
            tf.summary.histogram(name + '/deconv_output', h)
        h = instance_norm_op(h, reuse=reuse, name=name) if use_instance_norm else h
        h = batch_norm_op(h, fused=True, data_format='NCHW', renorm=use_batch_renorm) if (use_batch_norm or use_batch_renorm) else h
        if not reuse and use_batch_norm:
            tf.summary.scalar(name + '/batchnorm_output', tf.reduce_mean(h))
            tf.summary.histogram(name + '/batchnorm_output', h)
        h = activation(h) if activation else h
        if not reuse:
            tf.summary.scalar(name + '/activation_output', tf.reduce_mean(h))
            tf.summary.histogram(name + '/activation_output', h)
        h = tf.nn.dropout(h, keep_prob=dropout) if (dropout > 0) else h
        if not reuse:
            tf.add_to_collection('conv_layers', h)
    return h


# TODO simplfy this by calling hem.layers.conv2d twice, instead of inlining it here?. would need to add an option for returning the shortcut from conv2d
@add_arg_scope
def residual(x,
             input_size,
             output_size,
             filter_size=3,
             stride=1,
             init=xavier_initializer,
             use_batch_norm=False,
             use_batch_renorm=False,
             use_instance_norm=False,
             activation=None,
             reuse=False,
             dropout=0,
             padding='SAME',
             name=None):
    """A two-layer residual block.

    Args:
        x:
        input_size:
        output_size:
        filter_size:
        stride:
        init:
        use_batch_norm:
        use_batch_renorm:
        use_instance_norm:
        activation:
        reuse:
        dropout:
        padding:
        name:

    Returns:

    """
    with tf.name_scope(name, 'residual', [x]):
        w_name, b_name = _weight_name(name + 'A'), _bias_name(name + 'A')
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
        # conv layer 1
        strides = [1, 1, stride, stride]  # NCHW format
        h = tf.nn.conv2d(x, k, strides=strides, padding=padding, data_format='NCHW')
        h = tf.nn.bias_add(h, b, data_format='NCHW')
        # shortcut for later
        shortcut = tf.identity(h, 'shortcut')
        if not reuse:
            tf.summary.scalar(name + '/conv_outputA', tf.reduce_mean(h))
            tf.summary.histogram(name + '/conv_outputA', h)
        h = instance_norm_op(h, reuse=reuse, name=name) if use_instance_norm else h
        h = batch_norm_op(h, fused=True, data_format='NCHW', renorm=use_batch_renorm) if (use_batch_norm or use_batch_renorm) else h
        if not reuse and use_batch_norm:
            tf.summary.scalar(name + '/batchnorm_outputA', tf.reduce_mean(h))
            tf.summary.histogram(name + '/batchnorm_outputA', h)
        h = activation(h) if activation else h
        if not reuse:
            tf.summary.scalar(name + '/activationA', tf.reduce_mean(h))
            tf.summary.histogram(name + '/activationA', h)
        h = tf.nn.dropout(h, keep_prob=dropout) if (dropout > 0) else h
        if not reuse:
            tf.add_to_collection('residual_layers', h)

        # conv layer 2
        w_name, b_name = _weight_name(name + 'B'), _bias_name(name + 'B')
        with tf.variable_scope('vars', reuse=reuse):
            k = tf.get_variable(name=w_name,
                                shape=[filter_size, filter_size, output_size, output_size],
                                initializer=init())
            b = tf.get_variable(name=b_name,
                                shape=[output_size],
                                initializer=init())
            if not reuse:
                tf.add_to_collection('weights', k)
                tf.add_to_collection('biases', b)
        h = tf.nn.conv2d(h, k, strides=strides, padding=padding, data_format='NCHW')
        h = tf.nn.bias_add(h, b, data_format='NCHW')
        if not reuse:
            tf.summary.scalar(name + '/conv_outputB', tf.reduce_mean(h))
            tf.summary.histogram(name + '/conv_outputB', h)
        h = instance_norm_op(h, reuse=reuse, name=name) if use_instance_norm else h
        h = batch_norm_op(h, fused=True, data_format='NCHW', renorm=use_batch_renorm) if (use_batch_norm or use_batch_renorm) else h
        if not reuse and use_batch_norm:
            tf.summary.scalar(name + '/batchnorm_outputB', tf.reduce_mean(h))
            tf.summary.histogram(name + '/batchnorm_outputB', h)
        h = h + shortcut
        if not reuse:
            tf.summary.scalar(name + '/residual_output', tf.reduce_mean(h))
            tf.summary.histogram(name + '/residual_output', h)
        h = activation(h) if activation else h
        if not reuse:
            tf.summary.scalar(name + '/activationB', tf.reduce_mean(h))
            tf.summary.histogram(name + '/activationB', h)
        h = tf.nn.dropout(h, keep_prob=dropout) if (dropout > 0) else h

        if not reuse:
            tf.add_to_collection('residual_layers', h)

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


@add_arg_scope
def reshape(x,
            shape,
            name=None):
    """Reshapes the given tensor according to the desired data format.

    Convenience function.

    Args:
        x: Tensor
        shape: A shape in NHWC format.
        name: String,
    """
    shape = [shape[0], shape[3], shape[1], shape[2]]
    return tf.reshape(x, shape, name=name)

