"""Implementation of traditional convolutional autoencoder."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import hem
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
from math import sqrt


# TODO: build decoder from same weights as encoder

@hem.default_to_cpu
def cnn(x, args):
    """Initialize a standard convolutional autoencoder.

    Args:
      x: Tensor, input tensor representing the images.
      args: Argparse struct.
    """
    opt = hem.init_optimizer(args)
    tower_grads = []

    x = hem.rescale(x, (0,1), (-1,1))

    for x, scope, gpu_id in hem.tower_scope_range(x, args.n_gpus, args.batch_size):
        # create model
        with tf.variable_scope('encoder'):
            e = encoder(x, args, reuse=(gpu_id > 0))
        with tf.variable_scope('latent'):
            z = latent(e, args, reuse=(gpu_id > 0))
        with tf.variable_scope('decoder'):
            d = decoder(z, args, reuse=(gpu_id > 0))
        with tf.variable_scope('loss'):
            d_loss = loss(x, d, reuse=(gpu_id > 0))
        # compute gradients
        tower_grads.append(opt.compute_gradients(d_loss))
    # training
    avg_grads = hem.average_gradients(tower_grads)
    train_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())
    train_func = hem.default_training(train_op)
    # add summaries
    summaries(x, d, avg_grads, args)

    return train_func


def encoder(x, args, reuse=False):
    """Adds encoder nodes to the graph.

    Args:
      x: Tensor, input images.
      args: Argparse struct
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([hem.conv2d],
                       reuse = reuse,
                       activation = hem.lrelu):
        x = hem.conv2d(x,   3,  64, 5, 2, name='c1')
        x = hem.conv2d(x,  64, 128, 5, 2, name='c2')
        x = hem.conv2d(x, 128, 256, 5, 2, name='c3')
        x = hem.conv2d(x, 256, 256, 5, 2, name='c4')
        x = hem.conv2d(x, 256,  96, 1,    name='c5')
        x = hem.conv2d(x,  96,  32, 1,    name='c6')
    return x


def latent(x, args, reuse=False):
    """Add latant nodes to the graph.

    Args:
    x: Tensor, output from encoder.
    args: Argparse struct, containing:
               latent_size, integer describing size of the latent vector
    reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([hem.dense], reuse = reuse):
        x = hem.flatten(x)
        x = hem.dense(x, 32*4*4, args.latent_size, name='d1')
    return x


def decoder(x, args, reuse=False):
    """Adds decoder nodes to the graph.

    Args:
      x: Tensor, encoded image representation.
      args: Argparse struct
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([hem.dense, hem.conv2d, hem.deconv2d],
                       reuse = reuse,
                       activation = tf.nn.relu):
        x = hem.dense(x, args.latent_size, 32*4*4, name='d1')
        x = tf.reshape(x, [-1, 32, 4, 4])  # un-flatten
        x = hem.conv2d(x,    32,  96, 1,    name='c1')
        x = hem.conv2d(x,    96, 256, 1,    name='c2')
        x = hem.deconv2d(x, 256, 256, 5, 2, name='dc1')
        x = hem.deconv2d(x, 256, 128, 5, 2, name='dc2')
        x = hem.deconv2d(x, 128,  64, 5, 2, name='dc3')
        x = hem.deconv2d(x,  64,   3, 5, 2, name='dc4', activation=tf.nn.tanh)
    return x


def loss(x, d, reuse=False):
    """Add l1-loss nodes to the graph."""
    y = tf.reduce_mean(tf.abs(x - d), name='loss')
    if not reuse:
        tf.add_to_collection('losses', y)
    return y


def summaries(x, d, avg_grads, args):
    with tf.variable_scope('examples'):
        n = int(sqrt(args.examples))
        hem.montage(hem.rescale(x[0:args.examples], (-1, 1), (0, 1)), n, n, name='inputs')
        hem.montage(hem.rescale(d[0:args.examples], (-1, 1), (0, 1)), n, n, name='outputs')
    hem.summarize_activations()
    hem.summarize_losses()
    hem.summarize_weights_biases()
    hem.summarize_gradients(avg_grads)

