"""Implementation of variational autoencoder.

Sources:
-------
- Auto-Encoding Variational Bayes
https://arxiv.org/abs/1312.6114
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
from math import sqrt

from util import * #tower_scope_range, average_gradients, init_optimizer, default_to_cpu, merge_all_summaries, default_training
from ops.layers import dense, conv2d, deconv2d, flatten
from ops.summaries import montage_summary, summarize_gradients, summarize_activations, summarize_losses, summarize_weights_biases
from ops.activations import lrelu



@default_to_cpu
def vae(x, args):
    opt = init_optimizer(args)
    tower_grads = []

    for x, scope, gpu_id in tower_scope_range(x, args.n_gpus, args.batch_size):
        # model
        with tf.variable_scope('encoder'):
            e = encoder(x, reuse=(gpu_id>0))
        with tf.variable_scope('latent'):
            samples, z, z_mean, z_stddev = latent(e, args.batch_size, args.latent_size, reuse=(gpu_id>0))
        with tf.variable_scope('decoder'):
            d_real = decoder(z, args.latent_size, reuse=(gpu_id>0))
            d_fake = decoder(samples, args.latent_size, reuse=True)
        # losses
        d_loss, l_loss, t_loss = losses(x, z_mean, z_stddev, d_real)
        # gradients
        tower_grads.append(opt.compute_gradients(d_loss))

    # summaries
    summaries(x, d_fake, d_real, args)

    # training
    avg_grads = average_gradients(tower_grads)
    summarize_gradients(avg_grads)
    train_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())

    return default_training(train_op)



def summaries(x, d_fake, d_real, args):
    """Add montage summaries for examples and samples."""
    with tf.variable_scope('examples'):
        ne = int(sqrt(args.examples))
        montage_summary(x[0:args.examples], ne, ne, name='examples/inputs')
        montage_summary(d_real[0:args.examples], ne, ne, name='examples/real_decoded')
        montage_summary(d_fake[0:args.examples], ne, ne, name='examples/fake_decoded')
    summarize_activations()
    summarize_losses()
    summarize_weights_biases()
    
def losses(x, z_mean, z_stddev, d_real):
    """Adds loss nodes to the graph.

    Args:
      x: Tensor, real input images.
      z_mean: Tensor, latent's mean vector. 
      z_stddev: Tensor, latent's stddev vector.
      d_real: Tensor, output of decoder with real inputs.
    """
    # decoder loss
    exp = x * tf.log(1e-8 + d_real) + (1 - x) * tf.log(1e-8 + (1 - d_real))
    d_loss = tf.negative(tf.reduce_sum(exp), name='decoder_loss')
    
    # latent loss (using reparam trick)
    exp = tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-8 + tf.square(z_stddev)) - 1
    l_loss = tf.multiply(0.5, tf.reduce_sum(exp), name='latent_loss')
    # total loss
    t_loss = tf.reduce_mean(d_loss + l_loss, name='total_loss')
    # summaries
    for l in [d_loss, l_loss, t_loss]:
        tf.add_to_collection('losses', l)
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('l_loss', l_loss)
    tf.summary.scalar('t_loss', t_loss)
    return d_loss, l_loss, t_loss

    
def encoder(x, reuse=False):
    """Adds encoder nodes to the graph.

    Args:
      x: Tensor, input images.
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([conv2d],
                       reuse = reuse,
                       activation = lrelu,
                       use_batch_norm = True):
        x = conv2d(x, 3, 64, 5, 2, name='c1')
        x = conv2d(x, 64, 128, 5, 2, name='c2')
        x = conv2d(x, 128, 256, 5, 2, name='c3')
        x = conv2d(x, 256, 256, 5, 2, name='c4')
        x = conv2d(x, 256, 96, 1, name='c5')
        x = conv2d(x, 96, 32, 1, name='c6') #, activation=tf.nn.sigmoid)
    return x


def latent(x, batch_size, latent_size, reuse=False):
    """Adds latent nodes for sampling and reparamaterization.

    Args:
    x: Tensor, input images.
    batch_size: Integer, batch size.
    latent_size: Integer, size of latent vector.
    reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([dense],
                       reuse = reuse):
        flat = flatten(x)
        z_mean = dense(flat, 32*4*4, latent_size, name='d1')
        z_stddev = dense(flat, 32*4*4, latent_size, name='d2')
        samples = tf.random_normal([batch_size, latent_size], 0, 1)
        z = (z_mean + (z_stddev * samples))
    return (samples, z, z_mean, z_stddev)


def decoder(x, latent_size, reuse=False):
    """Adds decoder nodes to the graph.

    Args:
    x: Tensor, encoded image representation.
    latent_size: Integer, size of latent vector.
    reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([dense, conv2d, deconv2d],
                       activation = tf.nn.relu,
                       reuse = reuse):
        x = dense(x, latent_size, 32*4*4, name='d1')
        x = tf.reshape(x, [-1, 4, 4, 32])
        x = conv2d(x, 32, 96, 1, name='c1')
        x = conv2d(x, 96, 256, 1, name='c2')
        x = deconv2d(x, 256, 256, 5, 2, name='dc1')
        x = deconv2d(x, 256, 128, 5, 2, name='dc2')
        x = deconv2d(x, 128, 64, 5, 2, name='dc3')
        x = deconv2d(x, 64, 3, 5, 2, name='dc4', activation=tf.nn.sigmoid)
    return x

