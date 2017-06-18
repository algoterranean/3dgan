"""Implementation of traditional convolutional autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
from math import sqrt

from util import tower_scope_range, average_gradients, init_optimizer, default_to_cpu, merge_all_summaries, default_training
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
    x = 2 * (x - 0.5)

    for x, scope, gpu_id in tower_scope_range(x, args.n_gpus, args.batch_size):
        # create model
        with tf.variable_scope('encoder'):
            e = encoder(x, reuse=(gpu_id > 0))
        with tf.variable_scope('latent'):
            z = latent(e, args.latent_size, reuse=(gpu_id > 0))
        with tf.variable_scope('decoder'):
            d = decoder(z, args.latent_size, reuse=(gpu_id > 0))
        d_loss = loss(x, d)
        # add montage summaries
        summaries(x, d, args)
        # compute gradients
        tower_grads.append(opt.compute_gradients(d_loss))
                    
    # training
    avg_grads = average_gradients(tower_grads)
    summarize_gradients(avg_grads)
    train_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())

    return default_training(train_op), merge_all_summaries()



def summaries(x, d, args):
    ne = int(sqrt(args.examples))
    inputs = (x[0:args.examples] + 1.0) / 2.0
    outputs = (d[0:args.examples] + 1.0) / 2.0
    
    montage_summary(inputs, ne, ne, 'examples/inputs')
    montage_summary(outputs, ne, ne, 'examples/outputs')


def loss(x, d):
    """Add l1-loss nodes to the graph."""
    y = tf.reduce_mean(tf.abs(x - d), name='loss')
    tf.add_to_collection('losses', y)
    tf.summary.scalar('loss', y)
    tf.summary.histogram('loss', y)
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
                       activation = lrelu,
                       add_summaries = True):
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
                       add_summaries = True,
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




# class cnn:
#     """Implementation of standard convolutional auto-encoder."""
    
#     def __init__(self, x, args):
#         """Initialize model."""
#         with tf.device('/cpu:0'):
#             opt = init_optimizer(args)
#             tower_grads = []

#             for x, scope, gpu_id in tower_scope_range(x, args.n_gpus, args.batch_size):
#                 # model
#                 e = self.encoder(x, (gpu_id > 0))
#                 z = self.latent(e, args.latent_size, (gpu_id > 0))
#                 d = self.decoder(z, args.latent_size, (gpu_id > 0))
#                 loss = self.loss(x, d)
                
#                 # reuse variables on next tower
#                 tf.get_variable_scope().reuse_variables()
                
#                 # add montage summaries
#                 montage_summary(x, 8, 8, 'inputs')
#                 montage_summary(d, 8, 8, 'outputs')
                
#                 # compute gradients on this GPU
#                 tower_grads.append(opt.compute_gradients(loss))
                    
#             # back on the CPU
#             avg_grads = average_gradients(tower_grads)
#             summarize_gradients(avg_grads)
#             global_step = tf.train.get_global_step()
#             apply_grads = opt.apply_gradients(avg_grads, global_step=global_step)            
#             self.train_op = tf.group(apply_grads)

#             # grab summaries
#             summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
#             self.summary_op = tf.summary.merge(summaries)

            
#     def loss(self, x, d):
#         loss = tf.reduce_mean(tf.abs(x - d), name='loss')
#         tf.add_to_collectionn('losses', loss)
#         tf.summary.scalar('loss', loss)
#         tf.summary.histogram('loss', loss)
#         return loss


#     def latent(self, x, latent_size, reuse=False):
#         """Add latant nodes to the graph.

#         Args:
#           x: Tensor, output from encoder.
#           latent_size: Integer, size of latent vector.
#           reuse: Boolean, whether to reuse variables.
#         """
#         with arg_scope([dense], reuse = reuse):
#             x = flatten(x)
#             x = dense(x, 32*4*4, latent_size, name='d1')
#         return x
            

#     def encoder(self, x, reuse=False):
#         """Adds encoder nodes to the graph.

#         Args:
#           x: Tensor, input images.
#           reuse: Boolean, whether to reuse variables.
#         """
#         with arg_scope([conv2d],
#                            reuse = reuse,
#                            activation = lrelu,
#                            add_summaries = True):
#             x = conv2d(x,   3,  64, 5, 2, name='c1')
#             x = conv2d(x,  64, 128, 5, 2, name='c2')
#             x = conv2d(x, 128, 256, 5, 2, name='c3')
#             x = conv2d(x, 256, 256, 5, 2, name='c4')
#             x = conv2d(x, 256,  96, 1,    name='c5')
#             x = conv2d(x,  96,  32, 1,    name='c6')
#         return x

#     def decoder(self, x, latent_size, reuse=False):
#         """Adds decoder nodes to the graph.

#         Args:
#           x: Tensor, encoded image representation.
#           latent_size: Integer, size of latent vector.
#           reuse: Boolean, whether to reuse variables.
#         """
#         with arg_scope([dense, conv2d, deconv2d],
#                            reuse = reuse,
#                            add_summaries = True,
#                            activation = tf.nn.relu):
#             x = dense(x, latent_size, 32*4*4, name='d1')
#             x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
#             x = conv2d(x,    32,  96, 1,    name='c1')
#             x = conv2d(x,    96, 256, 1,    name='c2')
#             x = deconv2d(x, 256, 256, 5, 2, name='dc1')
#             x = deconv2d(x, 256, 128, 5, 2, name='dc2')
#             x = deconv2d(x, 128,  64, 5, 2, name='dc3')
#             x = deconv2d(x,  64,   3, 5, 2, name='dc4', activation=tf.nn.sigmoid)
#         return x


#     def train(self, sess, args):
#         sess.run(self.train_op)
