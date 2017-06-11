import tensorflow as tf
import numpy as np
import re
from math import sqrt
from tensorflow.contrib.layers import xavier_initializer as xav_init
from tensorflow.contrib.layers import variance_scaling_initializer as he_init
from tensorflow.contrib.layers import batch_norm


def weight_name(name):
    return name if name is None else name + '_w'


def bias_name(name):
    return name if name is None else name + '_b'


def lrelu(x, leak=0.2, name='lrelu'):
    """
    Leakly ReLU activation function.

    See: Rectifier Nonlinearities Improve Neural Network Acoustic Models
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    """
    return tf.maximum(leak*x, x, name=name)


def dense(x, input_size, output_size, reuse=False, name=None):
    """
    Standard dense (fully connected) layer.
    """
    w_name, b_name = weight_name(name), bias_name(name)
    with tf.variable_scope('vars', reuse=reuse):            
        W = tf.get_variable(name=w_name, shape=[input_size, output_size], initializer=xav_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=xav_init())
        tf.summary.histogram(w_name, W)
        tf.summary.histogram(b_name, b)
    return tf.matmul(x, W) + b
                        

def conv2d(x, input_size, output_size, ksize=3, stride=1, reuse=False, name=None, summary=False):
    """
    Standard 2D convolutional layer.
    """
    w_name, b_name = weight_name(name), bias_name(name)
    with tf.variable_scope('vars', reuse=reuse):
        K = tf.get_variable(name=w_name, shape=[ksize, ksize, input_size, output_size], initializer=xav_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=xav_init())
        tf.summary.histogram(w_name, K)
        tf.summary.histogram(b_name, b)
    h = tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME')
    y = tf.nn.bias_add(h, b)
    activation_summary(y)
    return y


def deconv2d(x, input_size, output_size, ksize=3, stride=2, reuse=False, name=None):
    """
    Standard fractionally strided (deconvolutional) layer.
    """
    w_name, b_name = weight_name(name), bias_name(name)
    with tf.variable_scope('vars', reuse=reuse):
        K = tf.get_variable(w_name, [ksize, ksize, output_size, input_size], initializer=xav_init())
        b = tf.get_variable(b_name, [output_size], initializer=xav_init())
        tf.summary.histogram(w_name, K)
        tf.summary.histogram(b_name, b)
    input_shape = tf.shape(x)
    output_shape = tf.stack([input_shape[0], input_shape[1]*2, input_shape[2]*2, output_size])
    h = tf.nn.conv2d_transpose(x, K, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')
    y = tf.nn.bias_add(h, b)
    activation_summary(y)
    return y


def flatten(x, name=None):
    """
    Flattens input tensor while preserving batch size.
    """
    output_size = tf.reduce_prod(tf.shape(x)[1:])
    return tf.reshape(x, [-1, output_size], name=name)


def input_slice(x, batch_size, gpu_id):
    """
    Returns a slice of x appropriate for this GPU (starting at gpu_id=0)
    """
    return x[gpu_id * batch_size : (gpu_id+1) * batch_size, :]



################################


def activation_summary(x, rows=0, cols=0, montage=True):
    n = re.sub('tower_[0-9]*/', '', x.op.name).split('/')[-1]
    tf.summary.histogram(n + '/activations/histogram', x)
    tf.summary.scalar(n + '/activations/sparsity', tf.nn.zero_fraction(x))
    if montage:
        montage_summary(tf.transpose(x[0], [2, 0, 1]), name=n + '/activations')


def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0:
            # if i == 1:
            #     print('PRIME?!')
            return (i, int(n/i))
        
        
def montage_summary(x, m=0, n=0, name='montage'):
    """
    Generates a m x n image montage from the given tensor.

    Will attempt to infer the input shape at run time but 
    if this is not possible, will need args m and n passed in.
    """
    if n == 0 or m == 0:
        m, n = factorization(x.get_shape()[0].value)
    else:
        num_examples = m * n
        
    images = tf.split(x, n, axis=0)
    images = tf.concat(images, axis=1)
    images = tf.unstack(images, m, axis=0)
    images = tf.concat(images, axis=1)
    # if this is a b/w images, add a third dimension
    if len(images.shape) < 3:
        images = tf.expand_dims(images, axis=2)
    # add a first dimension for expected TB format
    images = tf.expand_dims(images, axis=0)
    return tf.summary.image(name, images)



    
