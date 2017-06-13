"""Implementation of a few generative adversarial networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope

from util import tower_scope_range, average_gradients, init_optimizer, tensor_name
from model import Model
from ops.input import input_slice
from ops.layers import dense, conv2d, deconv2d, flatten
from ops.activations import lrelu
from ops.summaries import montage_summary, summarize_gradients



class GAN(Model):
    """Implementation of Vanilla GAN, Wasserstein GAN, and Improved
    Wasserstein GAN (denoted GAN, WGAN, and IWGAN, respectively).

    Which model is added to the graph is based on args.model ('gan',
    'wgan', or 'iwgan'), along with the other associated arguments
    (e.g. args.optimizer, args.latent_size, etc.) Supports multi-GPU
    training.

    Sources:
    -------
    - `Generative Adversarial Networks`
    https://arxiv.org/abs/1406.2661

    - `Wasserstein GAN`
    https://arxiv.org/abs/1701.07875

    - `Improved Training of Wasserstein GANs`
    https://arxiv.org/abs/1704.00028
    """
    
    def __init__(self, x, args):
        """Initialize model.
        
        Args:
          x: Tensor, the real images.
          args: Argparse structure.
        """

        with tf.device('/cpu:0'):
            g_opt, d_opt = init_optimizer(args), init_optimizer(args)
            g_tower_grads, d_tower_grads = [], []

            # flatten and rescale [0,1] to [-1,1]
            x = flatten(x)
            x = 2 * (x- 0.5)
            
            for x, scope, gpu_id in tower_scope_range(x, args.n_gpus, args.batch_size):
                # instantiate model on this GPU
                # x_slice = input_slice(x, args.batch_size, gpu_id)
                g = self.generator(args, reuse=(gpu_id>0))
                d_real = self.discriminator(x, args, reuse=(gpu_id>0))
                d_fake = self.discriminator(g, args, reuse=True)

                # reuse variables on next tower
                tf.get_variable_scope().reuse_variables()

                # losses
                g_loss, d_loss = self.losses(x, g, d_fake, d_real, args)

                # compute gradients
                g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_params))
                d_tower_grads.append(d_opt.compute_gradients(d_loss, var_list=d_params))

                # only need one batchnorm update (ends up being updates for last tower)
                batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                # add some summaries
                tf.summary.histogram('fakes', g)
                tf.summary.histogram('real', x)
                # add montages. need to rescale images from [-1,1] to [0,1]
                real_examples = (x[0:args.examples] + 1.0) / 2
                fake_examples = (g[0:args.examples] + 1.0) / 2
                montage_summary(tf.reshape(real_examples, [-1, 64, 64, 3]), 8, 8, 'inputs')
                montage_summary(tf.reshape(fake_examples, [-1, 64, 64, 3]), 8, 8, 'fake')

                
            # average gradients back on CPU
            g_grads = average_gradients(g_tower_grads)
            d_grads = average_gradients(d_tower_grads)
            summarize_gradients(g_grads + d_grads)

            # apply gradients back on CPU
            global_step = tf.train.get_global_step()
            g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=global_step)
            d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=global_step)
            
            # setup train ops
            if args.model == 'gan':
                with tf.control_dependencies(batchnorm_updates):
                    self.g_train_op = tf.group(g_apply_gradient_op)
                    self.d_train_op = tf.group(d_apply_gradient_op)
            elif args.model == 'iwgan':
                self.g_train_op = tf.group(g_apply_gradient_op)
                with tf.control_dependencies(batchnorm_updates):
                    self.d_train_op = tf.group(d_apply_gradient_op)
            elif args.model == 'wgan':
                # add weight clipping method
                clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_params]
                with tf.control_dependencies(batchnorm_updates):                
                    with tf.control_dependencies(clip_D):
                        self.d_train_op = tf.group(d_apply_gradient_op)
                    self.g_train_op = tf.group(g_apply_gradient_op)

        # grab summaries
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary_op = tf.summary.merge(summaries)




        
    def losses(self, x, g, d_fake, d_real, args):
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
        if args.model == 'gan':
            g_loss = tf.reduce_mean(-tf.log(d_fake + 1e-8))
            d_loss = tf.reduce_mean(-tf.log(d_real + 1e-8) - tf.log(1 - d_fake + 1e-8))
        elif args.model == 'wgan':
            g_loss = -tf.reduce_mean(d_fake)
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
        elif args.model == 'iwgan':
            l_term = 10.0
            g_loss = -tf.reduce_mean(d_fake)
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + l_term * self.gradient_penalty(x, g, args)
        tf.add_to_collection('losses', g_loss)
        tf.add_to_collection('losses', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_loss', d_loss)
        return g_loss, d_loss

        
    def gradient_penalty(self, x, g, args):
        """Calculate gradient penalty for discriminator.
        
        This requires creating a separate discriminator path.

        Args: 
          x: Tensor, the real images.
          g: Tensor, the fake images.
          args: Argparse structure.
        """
        alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0.0, maxval=1.0)
        differences = g - x
        interpolates = x + (alpha * differences)
        d_interpolates = self.discriminator(interpolates, args, reuse=True)
        gradients = tf.gradients(d_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
        return gradient_penalty


    def generator(self, args, reuse=False):
        """Adds generator nodes to the graph.

        From noise, applies deconv2d until image is scaled up to match
        the dataset.
        """
        output_dim = 64*64*3
        with tf.variable_scope('generator') as scope:
            with arg_scope([dense, deconv2d],
                               reuse = reuse,
                               use_batch_norm = True,
                               activation = tf.nn.relu,
                               add_summaries = True):
                z = tf.random_normal([args.batch_size, args.latent_size])
                y = dense(z, args.latent_size, 4*4*4*args.latent_size, name='fc1')
                y = tf.reshape(y, [-1, 4, 4, 4*args.latent_size])
                y = deconv2d(y, 4*args.latent_size, 2*args.latent_size, 5, 2, name='dc1')
                y = deconv2d(y, 2*args.latent_size, args.latent_size, 5, 2, name='dc2')
                y = deconv2d(y, args.latent_size, int(args.latent_size/2), 5, 2, name='dc3')
                y = deconv2d(y, int(args.latent_size/2), 3, 5, 2, name='dc4', activation=tf.tanh)
                y = tf.reshape(y, [-1, output_dim])
        return y

    
    def discriminator(self, x, args, reuse=False):
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
        use_bn = False if args.model == 'iwgan' else True
        with tf.variable_scope('discriminator') as scope:
            with arg_scope([conv2d],
                               use_batch_norm = use_bn,
                               activation = lrelu,
                               reuse = reuse,
                               add_summaries = True):
                x = tf.reshape(x, [-1, 64, 64, 3])
                x = conv2d(x, 3, args.latent_size, 5, 2, name='c1')
                x = conv2d(x, args.latent_size, args.latent_size*2, 5, 2, name='c2')
                x = conv2d(x, args.latent_size*2, args.latent_size*4, 5, 2, name='c3')
                x = tf.reshape(x, [-1, 4*4*4*args.latent_size])
                x = dense(x, 4*4*4*args.latent_size, 1, use_batch_norm=False, activation=None, name='fc2', reuse=reuse)
                x = tf.reshape(x, [-1])
        return x

    
    def train(self, sess, args):
        """Train the discriminator and generator.
        
        For WGAN and IWGAN, the discriminator is trained n times ("to
        convergence") before training the generator. 
        """
        if args.model == 'gan':
            sess.run([self.d_train_op, self.g_train_op])
        elif args.model in ['wgan', 'iwgan']:
            for i in range(args.n_disc_train):
                sess.run(self.d_train_op)
            sess.run(self.g_train_op)

