"""Implementation of variational autoencoder.

Sources:
-------
- Auto-Encoding Variational Bayes
https://arxiv.org/abs/1312.6114
"""

from __future__ import absolute_import, division, print_function

from math import sqrt

import tensorflow as tf
import hem
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope


@hem.default_to_cpu
def vae(x, args):
    opt = hem.init_optimizer(args)
    tower_grads = []

    for x, scope, gpu_id in hem.tower_scope_range(x, args.n_gpus, args.batch_size):
        # model
        with tf.variable_scope('encoder'):
            e = encoder(x, args, reuse=(gpu_id>0))
        with tf.variable_scope('latent'):
            samples, z, z_mean, z_stddev = latent(e, args.batch_size, args.latent_size, reuse=(gpu_id>0))
        with tf.variable_scope('decoder'):
            d_real = decoder(z, args, reuse=(gpu_id>0))
            d_fake = decoder(samples, args, reuse=True)
        # losses
        d_loss, l_loss, t_loss = losses(x, z_mean, z_stddev, d_real)
        # gradients
        tower_grads.append(opt.compute_gradients(d_loss))

    # summaries
    summaries(x, d_fake, d_real, args)

    # training
    avg_grads = hem.average_gradients(tower_grads)
    hem.summarize_gradients(avg_grads)
    train_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())

    return hem.default_training(train_op)



def summaries(x, d_fake, d_real, args):
    """Add montage summaries for examples and samples."""
    with tf.variable_scope('examples'):
        ne = int(sqrt(args.examples))
        hem.montage_summary(x[0:args.examples], ne, ne, name='examples/inputs')
        hem.montage_summary(d_real[0:args.examples], ne, ne, name='examples/real_decoded')
        hem.montage_summary(d_fake[0:args.examples], ne, ne, name='examples/fake_decoded')
    hem.summarize_activations()
    hem.summarize_losses()
    hem.summarize_weights_biases()
    
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

    
def encoder(x, args, reuse=False):
    """Adds encoder nodes to the graph.

    Args:
      x: Tensor, input images.
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([hem.conv2d],
                       reuse = reuse,
                       activation = hem.lrelu,
                       use_batch_norm = True):
        x = hem.conv2d(x, 3, 64, 5, 2, name='c1')
        x = hem.conv2d(x, 64, 128, 5, 2, name='c2')
        x = hem.conv2d(x, 128, 256, 5, 2, name='c3')
        x = hem.conv2d(x, 256, 256, 5, 2, name='c4')
        x = hem.conv2d(x, 256, 96, 1, name='c5')
        x = hem.conv2d(x, 96, 32, 1, name='c6') #, activation=tf.nn.sigmoid)
    return x


def latent(x, batch_size, latent_size, reuse=False):
    """Adds latent nodes for sampling and reparamaterization.

    Args:
    x: Tensor, input images.
    batch_size: Integer, batch size.
    latent_size: Integer, size of latent vector.
    reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([hem.dense],
                       reuse = reuse):
        flat = hem.flatten(x)
        z_mean = hem.dense(flat, 32*4*4, latent_size, name='d1')
        z_stddev = hem.dense(flat, 32*4*4, latent_size, name='d2')
        samples = tf.random_normal([batch_size, latent_size], 0, 1)
        z = (z_mean + (z_stddev * samples))
    return (samples, z, z_mean, z_stddev)


def decoder(x, args, reuse=False):
    """Adds decoder nodes to the graph.

    Args:
    x: Tensor, encoded image representation.
    latent_size: Integer, size of latent vector.
    reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([hem.dense, hem.conv2d, hem.deconv2d],
                       activation = tf.nn.relu,
                       reuse = reuse):
        x = hem.dense(x, args.latent_size, 32*4*4, name='d1')
        x = tf.reshape(x, [-1, 32, 4, 4])
        x = hem.conv2d(x, 32, 96, 1, name='c1')
        x = hem.conv2d(x, 96, 256, 1, name='c2')
        x = hem.deconv2d(x, 256, 256, 5, 2, name='dc1')
        x = hem.deconv2d(x, 256, 128, 5, 2, name='dc2')
        x = hem.deconv2d(x, 128, 64, 5, 2, name='dc3')
        x = hem.deconv2d(x, 64, 3, 5, 2, name='dc4', activation=tf.nn.sigmoid)
    return x

