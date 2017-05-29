# global
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import time
from sys import stdout
# local
from util import print_progress, fold, average_gradients, init_optimizer
from models.model import Model
from models.ops import dense, conv2d, deconv2d, lrelu, flatten, montage_summary, input_slice, activation_summary

relu = tf.nn.relu
var_scope = tf.variable_scope
name_scope = tf.name_scope


# batch norm discussion for multi-gpu training:
# https://github.com/tensorflow/tensorflow/issues/4361


class GAN(Model):
    """Vanilla Generative Adversarial Network with multi-GPU support.

    All variables are pinned to the CPU and shared between GPUs. 
    Gradients are collected on the CPU, averaged, and applied back to each GPU.
    We only keep one set of batch norm updates and include it as part of the train op.
    """
    
    def __init__(self, x, args, wgan=False):
        self.wgan = wgan
        
        with tf.device('/cpu:0'):
            # store optimizer and average grads on CPU
            g_opt = init_optimizer(args)
            d_opt = init_optimizer(args)
            g_tower_grads = []
            d_tower_grads = []

            # build model on each GPU
            with var_scope('model'):
                for gpu_id in range(args.n_gpus):
                    with tf.device(self.variables_on_cpu(gpu_id)):
                        with name_scope('tower_{}'.format(gpu_id)) as scope:
                            # get slice for this GPU
                            x_slice = input_slice(x, args.batch_size, gpu_id)
                            # build model  (note x is normalized to [-0.5, 0.5]
                            z, g, d_real, d_fake = self.build_model(x_slice - 0.5, args, gpu_id)
                            # generator (sample) summary
                            montage_summary(x_slice, 8, 8, 'inputs')
                            montage_summary(g, 8, 8, 'fake')
                            # TODO this depends on order in which it was added to collection... not ideal
                            # get losses and add summaries                            
                            g_loss, d_loss = self.summarize_collection('losses', scope)
                            # reuse variables in next tower
                            tf.get_variable_scope().reuse_variables()
                            # keep only one batchnorm update since one accumulates rapidly enough for both
                            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                            # compute and collect gradients for generator and discriminator
                            # restrict optimizer to vars for each component
                            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/generator')
                            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/discriminator')
                            with var_scope('compute_gradients/generator'):
                                g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_vars))
                            with var_scope('compute_gradients/discriminator'):
                                d_tower_grads.append(d_opt.compute_gradients(d_loss, var_list=d_vars))

            # back on the CPU
            with var_scope('training'):
                # average the gradients
                with var_scope('average_gradients/generator'):
                    g_grads = average_gradients(g_tower_grads)
                with var_scope('average_gradients/discriminator'):
                    d_grads = average_gradients(d_tower_grads)
                # apply the gradients
                with var_scope('apply_gradients/generator'):
                    g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=tf.train.get_global_step())
                with var_scope('apply_gradients/discriminator'):
                    d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=tf.train.get_global_step())
                # clip weights after each optimizer update
                with var_scope('clip_weights/discriminator'):
                    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
                # group training ops together
                with var_scope('train_op'):
                    batchnorm_updates_op = tf.group(*batchnorm_updates)
                    if self.wgan:
                        d_step = [d_apply_gradient_op, batchnorm_updates_op, *clip_D] * 5
                        self.train_op = tf.group(*d_step, g_apply_gradient_op)
                        # self.train_op = tf.group(d_apply_gradient_op, batchnorm_updates_op, *clip_D, g_apply_gradient_op)
                    else:
                        self.train_op = tf.group(d_apply_gradient_op, batchnorm_updates_op, g_apply_gradient_op)
                # summarize gradients
                self.summarize_grads(g_grads + d_grads)
                
            # combine all summary ops
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            self.summary_op = tf.summary.merge(summaries)

    
    def build_model(self, x, args, gpu_id):
        """Constructs the inference and loss portion of the model.
        Returns a list of form (latent, generator, real discriminator, 
        fake discriminator), where each is the final output node of each 
        component. """

        # latent
        with var_scope('latent'):
            z = self.build_latent(args.batch_size, args.latent_size)
        # generator
        with var_scope('generator'):
            g = self.build_generator(z, args.latent_size, (gpu_id > 0))
        # discriminator
        with var_scope('discriminator') as dscope:
            d_real = self.build_discriminator(x, (gpu_id>0))
            d_fake = self.build_discriminator(g, True)
        # losses
        self.build_losses(d_real, d_fake)
        
        return (z, g, d_real, d_fake)        

    
    def build_losses(self, d_real, d_fake):
        if self.wgan:
            with var_scope('losses/generator'):
                g_loss = tf.negative(tf.reduce_sum(d_fake), name='g_loss')
            with var_scope('losses/discriminator'):
                d_loss = tf.identity(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake), name='d_loss')
        else:
            # need to maximize this, but TF only does minimization, so we use negative        
            with var_scope('losses/generator'):
                g_loss = tf.reduce_mean(-tf.log(d_fake + 1e-8), name='g_loss')
            with var_scope('losses/discriminator'):
                d_loss = tf.reduce_mean(-tf.log(d_real + 1e-8) - tf.log(1 - d_fake + 1e-8), name='d_loss')
        tf.add_to_collection('losses', g_loss)
        tf.add_to_collection('losses', d_loss)
    
            
    def build_latent(self, batch_size, latent_size):
        """Builds a latent node that samples from the Gaussian."""
        
        z = tf.random_normal([batch_size, latent_size], 0, 1)
        activation_summary(z, montage=False)
        # sample node (for use in visualize.py)
        tf.identity(z, name='sample')
        return z

    
    def build_generator(self, x, batch_size, reuse):
        """Builds a generator with layers of size 
           96, 256, 256, 128, 64. Output is a 64x64x3 image."""
        
        with var_scope('fit_and_reshape', reuse=reuse):
            x = relu(batch_norm(dense(x, batch_size, 32*4*4, reuse, name='d1')))
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        with var_scope('conv1', reuse=reuse):
            b = conv2d(x, 32, 96, 1, reuse=reuse, name='c1')
            x = relu(batch_norm(b))
            activation_summary(x, 12, 8)
        with var_scope('conv2', reuse=reuse):                
            x = relu(batch_norm(conv2d(x, 96, 256, 1, reuse=reuse, name='c2')))
            activation_summary(x, 16, 16)
        with var_scope('deconv1', reuse=reuse):                
            x = relu(batch_norm(deconv2d(x, 256, 256, 5, 2, reuse=reuse, name='dc1')))
            activation_summary(x, 16, 16)
        with var_scope('deconv2', reuse=reuse):                
            x = relu(batch_norm(deconv2d(x, 256, 128, 5, 2, reuse=reuse, name='dc2')))
            activation_summary(x, 8, 16)
        with var_scope('deconv3', reuse=reuse):                
            x = relu(batch_norm(deconv2d(x, 128, 64, 5, 2, reuse=reuse, name='dc3')))
            activation_summary(x, 8, 8)
        with var_scope('deconv4', reuse=reuse): 
            x = relu(batch_norm(deconv2d(x, 64, 3, 5, 2, reuse=reuse, name='dc4')))
            activation_summary(x, montage=False)
        with var_scope('output'):
            x = tf.tanh(x, name='out')
            activation_summary(x, montage=False)
        # sample node (for use in visualize.py)            
        tf.identity(x, name='sample')
        return x
    

    def build_discriminator(self, x, reuse):
        """Builds a discriminator with layers of size
           64, 128, 256, 256, 96. Output is logits of final layer."""
        with var_scope('conv1', reuse=reuse):
            x = lrelu(conv2d(x, 3, 64, 5, 2, reuse=reuse, name='c1'))
            activation_summary(x, 8, 8)
        with var_scope('conv2', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 64, 128, 5, 2, reuse=reuse, name='c2')))
            activation_summary(x, 8, 16)
        with var_scope('conv3', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 128, 256, 5, 2, reuse=reuse, name='c3')))
            activation_summary(x, 16, 16)
        with var_scope('conv4', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 256, 256, 5, 2, reuse=reuse, name='c4')))
            activation_summary(x, 16, 16)
        with var_scope('conv5', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 256, 96, 32, 1, reuse=reuse, name='c5')))
            activation_summary(x, 12, 8)
        with var_scope('conv6', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 96, 32, 1, reuse=reuse, name='c6')))
            activation_summary(x, 8, 4)
        with var_scope('logits'):
            logits = flatten(x, name='flat1')
            if self.wgan:
                out = tf.identity(logits, name='out')
            else:
                out = tf.nn.sigmoid(logits, name='out')
            activation_summary(out, montage=False)
        # sample node (for use in visualize.py)
        tf.identity(out, name='sample')
        return out

        

