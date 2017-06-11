# global
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import time
import re
from sys import stdout
# local
from util import * 
from models.model import Model
from models.ops import *

TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
SUMMARIES = tf.GraphKeys.SUMMARIES
UPDATE_OPS = tf.GraphKeys.UPDATE_OPS



# Sources:
# - Wasserstein GAN
#   https://arxiv.org/abs/1701.07875
# 
# - Improved Training of Wasserstein GANs
#   https://arxiv.org/abs/1704.00028
# 
# - batch norm discussion for multi-gpu training:
#   https://github.com/tensorflow/tensorflow/issues/4361


# Example:
#
# python train.py --model wgan
#                 --data cifar
#                 --optimizer adam
#                 --beta1 0.5
#                 --beta2 0.9
#                 --lr 1e-4
#                 --batch_size 512
#                 --epochs 100
#                 --n_gpus 2
#                 --dir workspace/cifar_test
# 
# Note: Improved wgan paper used 200,000 iterations at batch size of 64,
#       and cifar-10 has 50,000 test images per epoch, for 256 total epochs.



class DCGAN(Model):
    """Vanilla Generative Adversarial Network with multi-GPU support.

    All variables are pinned to the CPU and shared between GPUs. 
    Gradients are collected on the CPU, averaged, and applied back to each GPU.
    We only keep one set of batch norm updates and include it as part of the train op.
    """
    
    def __init__(self, x, args, wgan=False):
        self.wgan = wgan
        global_step = tf.train.get_global_step()

        # x = tf.image.resize_images(x, [64, 64])
        x = tf.reshape(x, [-1, 32*32*3])
        x = 2*((x/255.0)-0.5)

        with tf.device('/cpu:0'):
            g_opt, d_opt = init_optimizer(args), init_optimizer(args)
            g_tower_grads, d_tower_grads = [], []


            # general model towers on each GPU
            for scope, gpu_id in tower_scope_range(args.n_gpus):
                # model and losses
                x_slice = input_slice(x, args.batch_size, gpu_id)
                z, g, d_real, d_fake = self.build_model(x_slice, args, gpu_id)
                g_loss, d_loss = tf.get_collection('losses', scope)

                # summaries
                montage_summary(tf.reshape(x_slice[0:args.examples], (-1, 32, 32, 3)), 'inputs')
                montage_summary(tf.reshape((g[0:args.examples]+1.0)/2.0, (-1, 32, 32, 3)), 8, 8, 'fake')
                summarize_collection('losses', scope)
                summaries = tf.get_collection(SUMMARIES, scope)

                # reuse variables in next tower
                tf.get_variable_scope().reuse_variables()
                
                # keep only one batchnorm update since one accumulates rapidly enough for both
                batchnorm_updates = tf.get_collection(UPDATE_OPS, scope)
                
                # compute and collect gradients for generator and discriminator
                # restrict optimizer to vars for each component
                g_vars = tf.get_collection(TRAINABLE_VARIABLES, scope='generator')
                d_vars = tf.get_collection(TRAINABLE_VARIABLES, scope='discriminator')
                with tf.variable_scope('compute_gradients/generator'):
                    g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_vars))
                with tf.variable_scope('compute_gradients/discriminator'):
                    d_tower_grads.append(d_opt.compute_gradients(d_loss, var_list=d_vars))

                

            # training
            with tf.variable_scope('training'):
                with tf.variable_scope('gradients'):
                    g_grads = average_gradients(g_tower_grads)
                    d_grads = average_gradients(d_tower_grads)
                    g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=global_step)
                    d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=global_step)
                    
                # group training ops together
                with tf.variable_scope('train_op'):
                    with tf.control_dependencies(batchnorm_updates):
                        self.train_disc_op = tf.group(d_apply_gradient_op)
                        self.train_gen_op = tf.group(g_apply_gradient_op)

            self.summary_op = tf.summary.merge(summaries)
                    

            # original wgan weight clipping method
            # if self.wgan:
            #     # clip weights after each optimizer update
            #     with tf.variable_scope('clip_weights/discriminator'):
            #         clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
            #
            # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)


    
    def build_model(self, x, args, gpu_id):
        """Constructs the inference and loss portion of the model.
        Returns a list of form (latent, generator, real discriminator, 
        fake discriminator), where each is the final output node of each 
        component. """

        # latent
        with tf.variable_scope('latent'):
            z = self.build_latent(args.batch_size, args.latent_size)
        # generator
        with tf.variable_scope('generator'):
            g = self.build_generator(z, args.batch_size, args.latent_size, (gpu_id > 0))
        # discriminator
        with tf.variable_scope('discriminator') as dscope:
            d_real = self.build_discriminator(x, (gpu_id>0)) #, not self.wgan)
            d_fake = self.build_discriminator(g, True) #, not self.wgan)

            # calculate gradient penalty term (from `Improved Training of Wasserstein GANs`)
            differences = g - x
            alpha = tf.random_uniform(shape=[args.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            interpolates = x + alpha*differences
            d_interpolates = self.build_discriminator(interpolates, True) #, not self.wgan)
            gradients = tf.gradients(d_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        # losses
        self.build_losses(x, g, d_real, d_fake, gradient_penalty, gpu_id, args)
        
        return (z, g, d_real, d_fake)

    
    def build_losses(self, x, g, d_real, d_fake, gradient_penalty, gpu_id, args):
        l_term = 10.0
        g_loss = tf.negative(tf.reduce_mean(d_fake), name='g_loss')
        d_loss = tf.identity(tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + l_term * gradient_penalty, name='d_loss')
        
        # if self.wgan:
        #     with tf.variable_scope('losses/generator'):
        #         g_loss = tf.negative(tf.reduce_mean(d_fake), name='g_loss')
        #     with tf.variable_scope('losses/discriminator'):
        #         l_term = 10.0 # TODO move to args
        #         d_loss = tf.identity(tf.reduce_mean(d_fake) - \
        #                                  tf.reduce_mean(d_real) + \
        #                                  l_term * gradient_penalty, name='d_loss')
        # else:
        #     # need to maximize this, but TF only does minimization, so we use negative
        #     with tf.variable_scope('losses/generator'):
        #         g_loss = tf.reduce_mean(-tf.log(d_fake + 1e-8), name='g_loss')
        #     with tf.variable_scope('losses/discriminator'):
        #         d_loss = tf.reduce_mean(-tf.log(d_real + 1e-8) - tf.log(1 - d_fake + 1e-8), name='d_loss')
        tf.add_to_collection('losses', g_loss)
        tf.add_to_collection('losses', d_loss)
    
            
    def build_latent(self, batch_size, latent_size):
        """Builds a latent node that samples from the Gaussian."""
        
        # z = tf.random_normal([batch_size, latent_size], 0, 1)
        z = tf.random_normal([batch_size, 128], 0, 1)        
        # activation_summary(z, montage=False)
        # # sample node (for use in visualize.py)
        # tf.identity(z, name='sample')
        return z

    
    def build_generator(self, x, batch_size, latent_size, reuse, use_batch_norm=True):
        """Builds a generator with layers of size 
           96, 256, 256, 128, 64. Output is a 64x64x3 image."""
        DIM = 128
        x = tf.nn.relu(batch_norm(dense(x, 128, 4*4*4*DIM, reuse=reuse, name='fc1')))
        x = tf.reshape(x, [-1, 4, 4, 4*DIM])
        x = tf.nn.relu(batch_norm(deconv2d(x, 4*DIM, 2*DIM, 5, 2, reuse=reuse, name='dc1')))
        x = tf.nn.relu(batch_norm(deconv2d(x, 2*DIM, DIM, 5, 2, name='dc2')))
        x = tf.tanh(deconv2d(x, DIM, 3, 5, 2, name='dc3'))
        return tf.reshape(x, [-1, 32*32*3])
        
        

        # x = tf.nn.relu(batch_norm(dense(x, latent_size, 16384, reuse, name='fc1')))
        # x = tf.reshape(x, [-1, 4, 4, 1024])
        # x = tf.nn.relu(batch_norm(deconv2d(x, 1024, 512, 5, 2, reuse=reuse, name='dc1')))
        # x = tf.nn.relu(batch_norm(deconv2d(x, 512, 256, 5, 2, reuse=reuse, name='dc2')))
        # x = tf.nn.relu(batch_norm(deconv2d(x, 256, 128, 5, 2, reuse=reuse, name='dc3')))
        # x = tf.nn.relu(batch_norm(deconv2d(x, 128, 3, 5, 2, reuse=reuse, name='dc4')))
        # x = tf.tanh(x)
        # return x

        # with tf.variable_scope('fit_and_reshape', reuse=reuse):
        #     x = tf.nn.relu(batch_norm(dense(x, batch_size, 32*4*4, reuse, name='d1'), center=True, scale=True, is_training=True))
        #     x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        # with tf.variable_scope('conv1', reuse=reuse):
        #     x = tf.nn.relu(batch_norm(conv2d(x, 32, 96, 1, reuse=reuse, name='c1'), center=True, scale=True, is_training=True))
        #     activation_summary(x, 12, 8)
        # with tf.variable_scope('conv2', reuse=reuse):                
        #     x = tf.nn.relu(batch_norm(conv2d(x, 96, 256, 1, reuse=reuse, name='c2'), center=True, scale=True, is_training=True))
        #     activation_summary(x, 16, 16)
        # with tf.variable_scope('deconv1', reuse=reuse):                
        #     x = tf.nn.relu(batch_norm(deconv2d(x, 256, 256, 5, 2, reuse=reuse, name='dc1'), center=True, scale=True, is_training=True))
        #     activation_summary(x, 16, 16)
        # with tf.variable_scope('deconv2', reuse=reuse):                
        #     x = tf.nn.relu(batch_norm(deconv2d(x, 256, 128, 5, 2, reuse=reuse, name='dc2'), center=True, scale=True, is_training=True))
        #     activation_summary(x, 8, 16)
        # with tf.variable_scope('deconv3', reuse=reuse):                
        #     x = tf.nn.relu(batch_norm(deconv2d(x, 128, 64, 5, 2, reuse=reuse, name='dc3'), center=True, scale=True, is_training=True))
        #     activation_summary(x, 8, 8)
        # with tf.variable_scope('deconv4', reuse=reuse):
        #     # 32x32
        #     x = tf.nn.relu(batch_norm(conv2d(x, 64, 3, 1, reuse=reuse, name='c3'), center=True, scale=True, is_training=True))
        #     # 64x64
        #     # x = tf.nn.relu(batch_norm(deconv2d(x, 64, 3, 5, 2, reuse=reuse, name='dc4')))
        #     activation_summary(x, montage=False)
        # with tf.variable_scope('output'):
        #     x = tf.tanh(x, name='out')
        #     activation_summary(x, montage=False)
            
        # # sample node (for use in visualize.py)            
        # tf.identity(x, name='sample')
        # return x
    

    def build_discriminator(self, x, reuse, use_batch_norm=True):
        """Builds a discriminator with layers of size
           64, 128, 256, 256, 96. Output is logits of final layer."""

        DIM = 128
        x = tf.reshape(x, [-1, 32, 32, 3])
        x = lrelu(conv2d(x, 3, DIM, 5, 2, reuse=reuse, name='c1'))
        x = lrelu(conv2d(x, DIM, 2*DIM, 5, 2, reuse=reuse, name='c2'))
        x = lrelu(conv2d(x, 2*DIM, 4*DIM, 5, 2, reuse=reuse, name='c3'))
        x = flatten(x)
        x = dense(x, 4*4*4*DIM, 1, reuse=reuse, name='fc2')
        return tf.reshape(x, [-1])

        # x = lrelu(conv2d(x, 3, 64, 5, 2, reuse=reuse, name='c1'))
        # x = lrelu(conv2d(x, 64, 128, 5, 2, reuse=reuse, name='c2'))
        # x = lrelu(conv2d(x, 128, 256, 5, 2, reuse=reuse, name='c3'))
        # x = lrelu(conv2d(x, 256, 512, 5, 2, reuse=reuse, name='c4'))
        # x = flatten(x)
        # x = dense(x, 8192, 1, reuse=reuse, name='fc2')
        # # x = flatten(x)
        # # x = lrelu(dense(x, 2048, 1, reuse=reuse, name='fc2'))
        # # x = tf.nn.sigmoid(x)
        # return x

        
        # with tf.variable_scope('conv1', reuse=reuse):
        #     x = lrelu(conv2d(x, 3, 64, 5, 2, reuse=reuse, name='c1'))
        #     activation_summary(x, 8, 8)
        # with tf.variable_scope('conv2', reuse=reuse):
        #     x = conv2d(x, 64, 128, 5, 2, reuse=reuse, name='c2')
        #     if use_batch_norm:
        #         x = batch_norm(x, center=True, scale=True, is_training=True)
        #     x = lrelu(x)
        #     activation_summary(x, 8, 16)
        # with tf.variable_scope('conv3', reuse=reuse):
        #     x = conv2d(x, 128, 256, 5, 2, reuse=reuse, name='c3')
        #     if use_batch_norm:
        #         x = batch_norm(x, center=True, scale=True, is_training=True)
        #     x = lrelu(x)
        #     activation_summary(x, 16, 16)
        # with tf.variable_scope('conv4', reuse=reuse):
        #     x = conv2d(x, 256, 256, 5, 2, reuse=reuse, name='c4')
        #     if use_batch_norm:
        #         x = batch_norm(x, center=True, scale=True, is_training=True)
        #     x = lrelu(x)
        #     activation_summary(x, 16, 16)
        # with tf.variable_scope('conv5', reuse=reuse):
        #     x = conv2d(x, 256, 96, 32, 1, reuse=reuse, name='c5')
        #     if use_batch_norm:
        #         x = batch_norm(x, center=True, scale=True, is_training=True)
        #     x = lrelu(x)
        #     activation_summary(x, 12, 8)
        # with tf.variable_scope('conv6', reuse=reuse):
        #     x = conv2d(x, 96, 32, 1, reuse=reuse, name='c6')
        #     if use_batch_norm:
        #         x = batch_norm(x, center=True, scale=True, is_training=True)
        #     x = lrelu(x)
        #     activation_summary(x, 8, 4)

        # # TODO add fully connected layer to logits 
            
        # with tf.variable_scope('logits'):
        #     logits = flatten(x, name='flat1')
        #     if self.wgan:
        #         out = tf.identity(logits, name='out')
        #     else:
        #         out = tf.nn.sigmoid(logits, name='out')
        #     activation_summary(out, montage=False)
        # # sample node (for use in visualize.py)
        # tf.identity(out, name='sample')
        # return out


    def train(self, sess, args):
        for i in range(args.n_disc_train):
            sess.run(self.train_disc_op)
        sess.run(self.train_gen_op)
        

