"""Implementation of Conditional GAN. Uses Improved Wasserstein methods. 

Supports multi-GPU training.

Sources:
-------
- `Image-to-Image Translation with Conditional Adversarial Networks`
  http://arxiv.org/abs/1611.07004
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope

from util import * #tower_scope_range, average_gradients, init_optimizer, default_to_cpu, merge_all_summaries
from ops.layers import dense, conv2d, deconv2d, flatten
from ops.activations import lrelu
from ops.summaries import montage_summary, summarize_gradients, summarize_activations, summarize_losses, summarize_weights_biases



@default_to_cpu
def cgan(x, args):
    """Initialize model.
        
    Args:
    x: Tensor, the real images.
    args: Argparse structure.
    """
    g_opt, d_opt = init_optimizer(args), init_optimizer(args)
    g_tower_grads, d_tower_grads = [], []

    # x = flatten(x)
    x = 2 * (x - 0.5)
    
    for x, scope, gpu_id in tower_scope_range(x, args.n_gpus, args.batch_size):
        # model
        with tf.variable_scope('generator'):
            e = encoder(x[:,:,:,0:3], reuse=(gpu_id>0))
            d = decoder(e, args.latent_size, args.batch_size, reuse=(gpu_id>0))
            g = tf.concat((x[:,:,:,0:3], d), axis=3)
            # g = generator(x[:,:,:,0:3], args.batch_size, args.latent_size, args, reuse=(gpu_id>0))
        with tf.variable_scope('discriminator'):
            d_real = discriminator(x, args, reuse=(gpu_id>0))
            d_fake = discriminator(g, args, reuse=True)
        # losses
        g_loss, d_loss = losses(x, g, d_fake, d_real, args)
        # compute gradients
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_params))
        d_tower_grads.append(d_opt.compute_gradients(d_loss, var_list=d_params))
        # only need one batchnorm update (ends up being updates for last tower)
        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
        
    # summaries
    summaries(g, x, args)
        
    # average and apply gradients
    g_grads = average_gradients(g_tower_grads)
    d_grads = average_gradients(d_tower_grads)
    summarize_gradients(g_grads + d_grads)
    global_step = tf.train.get_global_step()
    g_apply_grads = g_opt.apply_gradients(g_grads, global_step=global_step)
    d_apply_grads = d_opt.apply_gradients(d_grads, global_step=global_step)
    
    # training
    train_func = _train_cgan(g_apply_grads, d_apply_grads, batchnorm_updates)

        
    return train_func


def summaries(g, x, args):
    """Adds histogram and montage summaries for real and fake examples."""
    with tf.variable_scope('examples'):
        tf.summary.histogram('fakes', g)
        tf.summary.histogram('real', x)
        with tf.variable_scope('rescale'):
            # need to rescale images from [-1,1] to [0,1]
            real_examples = (x[0:args.examples] + 1.0) / 2
            fake_examples = (g[0:args.examples] + 1.0) / 2
        montage_summary(tf.reshape(real_examples[:,:,:,0:3], [-1, 64, 64, 3]), 8, 8, 'input_images')
        montage_summary(tf.reshape(real_examples[:,:,:,3], [-1, 64, 64, 1]), 8, 8, 'input_depths')
        montage_summary(tf.reshape(fake_examples[:,:,:,0:3], [-1, 64, 64, 3]), 8, 8, 'fake_images')
        montage_summary(tf.reshape(fake_examples[:,:,:,3], [-1, 64, 64, 1]), 8, 8, 'fake_depths')
        tf.summary.histogram('real_red_values', x[:,:,:,0])
        tf.summary.histogram('real_green_values', x[:,:,:,1])
        tf.summary.histogram('real_blue_values', x[:,:,:,2])
        tf.summary.histogram('real_depth_values', x[:,:,:,3])
        tf.summary.histogram('fake_red_values', g[:,:,:,0])
        tf.summary.histogram('fake_green_values', g[:,:,:,1])
        tf.summary.histogram('fake_blue_values', g[:,:,:,2])
        tf.summary.histogram('fake_depth_values', g[:,:,:,3])
    summarize_activations()
    summarize_losses()
    summarize_weights_biases()


def _train_cgan(g_apply_grads, d_apply_grads, batchnorm_updates):
    """Generates helper to train a Conditional IWGAN.

    Batchnorm updates are applies only to discriminator, since generator
    doesn't use batchnorm. Discriminator is trained to convergence
    before training the generator.
    """
    g_train_op = g_apply_grads
    with tf.control_dependencies(batchnorm_updates):
        d_train_op = d_apply_grads
    losses = collection_to_dict(tf.get_collection('losses'))
    def helper(sess, args):
        for i in range(args.n_disc_train):
            sess.run(d_train_op)
        _, l = sess.run([g_train_op, losses])
        return l
        
    return helper

        
def losses(x, g, d_fake, d_real, args):
    """Add loss nodes to the graph depending on the model type.

    Args:
      x: Tensor, real images.
      g: Tensor, the generator.
      d_fake: Tensor, the fake discriminator.
      d_real: Tensor, the real discriminator.
      args: Argparse structure.

    Returns:
      g_loss: Tensor, the generator's loss function.
      d_loss: Tensor, the discriminator's loss function.
    """
    l_term = 10.0
    rmse_loss = tf.sqrt(tf.reduce_mean(tf.square((x[:,:,:,3]+1.0)/2.0 - (g[:,:,:,3]+1.0)/2.0)), name='rmse_loss')
    g_loss = tf.identity(-tf.reduce_mean(d_fake) + rmse_loss, name='g_loss')
    with tf.variable_scope('discriminator'):
        gp = gradient_penalty(x, g, args)
    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + l_term * gp
    d_loss = tf.identity(d_loss, name='d_loss')
    
    # rmse_loss = tf.sqrt(tf.reduce_mean(tf.square((x[:,:,:,3]+1.0)/2.0 - (g[:,:,:,3]+1.0)/2.0)), name='rmse_loss')
    
    tf.add_to_collection('losses', g_loss)
    tf.add_to_collection('losses', d_loss)
    tf.add_to_collection('losses', rmse_loss)
    return g_loss, d_loss


def gradient_penalty(x, g, args):
    """Calculate gradient penalty for discriminator.
        
    This requires creating a separate discriminator path.

    Args: 
    x: Tensor, the real images.
    g: Tensor, the fake images.
    args: Argparse structure.
    """
    x = flatten(x)
    g = flatten(g)
    
    alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0.0, maxval=1.0)
    differences = g - x
    interpolates = x + (alpha * differences)
    interpolates = tf.reshape(interpolates, [-1, 64, 64, 4])
    d_interpolates = discriminator(interpolates, args, reuse=True)
    gradients = tf.gradients(d_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
    penalty = tf.reduce_mean((slopes-1.0)**2)
    return penalty


def encoder(x, reuse=False):
    """Adds encoder nodes to the graph.

    Args:
      x: Tensor, input images.
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([conv2d],
                       reuse = reuse,
                       activation = lrelu):
        x = conv2d(x,   3,  64, 5, 2, name='e_c1')
        x = conv2d(x,  64, 128, 5, 2, name='e_c2')
        x = conv2d(x, 128, 256, 5, 2, name='e_c3')
        x = conv2d(x, 256, 256, 5, 2, name='e_c4')
        x = conv2d(x, 256,  96, 1,    name='e_c5')
        x = conv2d(x,  96,  32, 1,    name='e_c6')
    return x


def decoder(x, latent_size, batch_size, reuse=False):
    """Adds decoder nodes to the graph.

    Args:
      x: Tensor, encoded image representation.
      latent_size: Integer, size of latent vector.
      reuse: Boolean, whether to reuse variables.
    """
    with arg_scope([dense, conv2d, deconv2d],
                       reuse = reuse,
                       activation = tf.nn.relu):
        # x = flatten(x)
        # x = dense(x, 32*4*4, latent_size, name='d_d1')
        # x = dense(x, latent_size, 32*4*4, name='d_d2')
        # x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        x = conv2d(x,    32,  96, 1,    name='d_c1')
        x = conv2d(x,    96, 256, 1,    name='d_c2')
        x = deconv2d(x, 256, 256, 5, 2, name='d_dc1')
        x = deconv2d(x, 256, 128, 5, 2, name='d_dc2')
        x = deconv2d(x, 128,  64, 5, 2, name='d_dc3')
        x = deconv2d(x,  64,   1, 5, 2, name='d_dc4', activation=tf.nn.tanh)
    return x



# def generator(x, batch_size, latent_size, args, reuse=False):
#     """Adds generator nodes to the graph.

#     From noise, applies deconv2d until image is scaled up to match
#     the dataset.
#     """
#     # x = x[:,0:64*64*3] # image
#     output_dim = 64*64*4
#     with arg_scope([dense, deconv2d],
#                        reuse = reuse,
#                        use_batch_norm = True,
#                        activation = tf.nn.relu):
        
#         # z = tf.random_normal([batch_size, latent_size])
#         # # x = flatten(x)
#         # z = tf.concat((z, flatten(x)), axis=1)
#         y = dense(x, latent_size+64*64*3, 4*4*4*latent_size, name='fc1')
#         y = tf.reshape(y, [-1, 4, 4, 4*latent_size])
#         y = deconv2d(y, 4*latent_size, 2*latent_size, 5, 2, name='dc1')
#         y = deconv2d(y, 2*latent_size, latent_size, 5, 2, name='dc2')
#         y = deconv2d(y, latent_size, int(latent_size/2), 5, 2, name='dc3')
#         y = deconv2d(y, int(latent_size/2), 1, 5, 2, name='dc4', activation=tf.tanh, use_batch_norm=False)
#         y = tf.concat((x, y), axis=3)
#         # y = tf.reshape(y, [-1, output_dim])
#     return y


def discriminator(x, args, reuse=False):
    """Adds discriminator nodes to the graph.

    From the input image, successively applies convolutions with
    striding to scale down layer sizes until we get to a single
    output value, representing the discriminator's estimate of fake
    vs real. The single final output acts similar to a sigmoid
    activation function.

    Args:
    x: Tensor, input.
    args: Argparse structure.
    reuse: Boolean, whether to reuse variables.

    Returns:
    Final output of discriminator pipeline.
    """
    use_bn = False
    final_activation = None
    with arg_scope([conv2d],
                       use_batch_norm = use_bn,
                       activation = lrelu,
                       reuse = reuse):
        # x = tf.reshape(x, [-1, 64, 64, 4])
        x = conv2d(x, 4, args.latent_size, 5, 2, name='c1', use_batch_norm=False)
        x = conv2d(x, args.latent_size, args.latent_size*2, 5, 2, name='c2')
        x = conv2d(x, args.latent_size*2, args.latent_size*4, 5, 2, name='c3')
        x = tf.reshape(x, [-1, 4*4*4*args.latent_size])
        x = dense(x, 4*4*4*args.latent_size, 1, use_batch_norm=False, activation=final_activation, name='fc2', reuse=reuse)
        x = tf.reshape(x, [-1])
    return x
