"""Implementation of traditional convolutional autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
from math import sqrt

from util import * #tower_scope_range, average_gradients, init_optimizer, default_to_cpu, merge_all_summaries, default_training, tensor_name
from ops.layers import dense, conv2d, deconv2d, flatten
from ops.summaries import montage_summary, summarize_gradients
from ops.activations import lrelu


# TODO: build decoder from same weights as encoder

@default_to_cpu
def cnn(x, args):
    """Initialize a standard convolutional autoencoder.

    Args:
      x: Tensor, input tensor representing the images.
      args: Argparse struct. 
    """
    opt = init_optimizer(args)
    tower_grads = []

    # rescale to [-1, 1]
    with tf.variable_scope('rescale'):
        x = 2 * (x - 0.5)

    for x, scope, gpu_id in tower_scope_range(x, args.n_gpus, args.batch_size):
        # create model
        with tf.variable_scope('encoder'):
            e = encoder(x, reuse=(gpu_id > 0))
        with tf.variable_scope('latent'):
            z = latent(e, args.latent_size, reuse=(gpu_id > 0))
        with tf.variable_scope('decoder'):
            d = decoder(z, args.latent_size, reuse=(gpu_id > 0))
        with tf.variable_scope('loss'):
            d_loss = loss(x, d)
        # compute gradients
        tower_grads.append(opt.compute_gradients(d_loss))


    summaries(x, d, args)
                    
    # training
    avg_grads = average_gradients(tower_grads)
    summarize_gradients(avg_grads)
    train_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())

    # col = tf.get_collection('layers')
    # for l in col:
    #     print(l)

    return default_training(train_op), merge_all_summaries()


def summaries(x, d, args):
    with tf.variable_scope('examples'):
        with tf.variable_scope('rescale'):
            x_scaled = (x + 1.0) / 2.0
            d_scaled = (d + 1.0) / 2.0
        ne = int(sqrt(args.examples))
        montage_summary(x_scaled[0:args.examples], ne, ne, 'inputs')
        montage_summary(d_scaled[0:args.examples], ne, ne, 'outputs')

    with tf.variable_scope('activations'):
        for l in tf.get_collection('conv_layers'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))
            montage_summary(tf.transpose(l[0], [2, 0, 1]), name=tensor_name(l) + '/montage')
        for l in tf.get_collection('dense_layers'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))

    with tf.variable_scope('loss'):
        for l in tf.get_collection('losses'):
            tf.summary.scalar(tensor_name(l), l)
            tf.summary.histogram(tensor_name(l), l)

    with tf.variable_scope('weights'):
        for l in tf.get_collection('weights'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))
            # montage_summary(l, name=tensor_name(l) + '/montage')

    with tf.variable_scope('biases'):
        for l in tf.get_collection('biases'):
            tf.summary.histogram(tensor_name(l), l)
            tf.summary.scalar(tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))


def loss(x, d):
    """Add l1-loss nodes to the graph."""
    y = tf.reduce_mean(tf.abs(x - d), name='loss')
    tf.add_to_collection('losses', y)
    return y


def latent(x, latent_size, reuse=False):
    """Add latant nodes to the graph.

    Args:
    x: Tensor, output from encoder.
    latent_size: Integer, size of latent vector.
    reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([dense], reuse = reuse):
        x = flatten(x)
        x = dense(x, 32*4*4, latent_size, name='d1')
    return x
            

def encoder(x, reuse=False):
    """Adds encoder nodes to the graph.

    Args:
      x: Tensor, input images.
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([conv2d],
                       reuse = reuse,
                       activation = lrelu):
        x = conv2d(x,   3,  64, 5, 2, name='c1')
        x = conv2d(x,  64, 128, 5, 2, name='c2')
        x = conv2d(x, 128, 256, 5, 2, name='c3')
        x = conv2d(x, 256, 256, 5, 2, name='c4')
        x = conv2d(x, 256,  96, 1,    name='c5')
        x = conv2d(x,  96,  32, 1,    name='c6')
    return x


def decoder(x, latent_size, reuse=False):
    """Adds decoder nodes to the graph.

    Args:
      x: Tensor, encoded image representation.
      latent_size: Integer, size of latent vector.
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([dense, conv2d, deconv2d],
                       reuse = reuse,
                       activation = tf.nn.relu):
        x = dense(x, latent_size, 32*4*4, name='d1')
        x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        x = conv2d(x,    32,  96, 1,    name='c1')
        x = conv2d(x,    96, 256, 1,    name='c2')
        x = deconv2d(x, 256, 256, 5, 2, name='dc1')
        x = deconv2d(x, 256, 128, 5, 2, name='dc2')
        x = deconv2d(x, 128,  64, 5, 2, name='dc3')
        x = deconv2d(x,  64,   3, 5, 2, name='dc4', activation=tf.nn.tanh)
    return x

