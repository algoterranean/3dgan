import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import time
import re
from sys import stdout
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, M
from util import print_progress, fold, average_gradients, init_optimizer




class GAN(Model):
    """Vanilla Generative Adversarial Network with multi-GPU support.

    All variables are pinned to the CPU and shared between GPUs. 
    Gradients are collected on the CPU, averaged, and applied back to each GPU."""


    def __init__(self, x, global_step, args):
        with tf.device('/cpu:0'):
            # store optimizer and grads on CPU
            with tf.variable_scope('optimizers'):
                with tf.variable_scope('generator'):
                    g_opt = init_optimizer(args)
                with tf.variable_scope('disciminator'):
                    d_opt = init_optimizer(args)

            g_tower_grads = []
            d_tower_grads = []

            # build model on each GPU
            with tf.variable_scope('model'): # tf.get_variable_scope()):
                for gpu_id in range(args.n_gpus):
                    with tf.device('/gpu:{}'.format(gpu_id)):
                        with tf.name_scope('TOWER_{}'.format(gpu_id)) as scope:
                            # build model
                            z, g, d_real, d_fake = self.construct_model(x, args, gpu_id)
                            # get losses and add summaries
                            losses = tf.get_collection('losses', scope)
                            for l in losses:
                                tf.summary.scalar(self.tensor_name(l), l)
                            g_loss, d_loss = losses

                            # reuse variables in next tower
                            tf.get_variable_scope().reuse_variables()

                            # add summaries for generated images and real images
                            tf.summary.image('real_images', x)
                            tf.summary.image('fake_images', g)

                            # use the last tower's summaries
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # compute and collect gradients for generator and discriminator
                            # restrict optimizer to vars for each component
                            params = tf.trainable_variables()
                            g_vars = [i for i in params if 'generator' in i.name]
                            d_vars = [i for i in params if 'discriminator' in i.name]
                            with tf.variable_scope('compute_gradients'):
                                with tf.variable_scope('generator'):
                                    g_grads = g_opt.compute_gradients(g_loss, g_vars)
                                with tf.variable_scope('discriminator'):
                                    d_grads = d_opt.compute_gradients(d_loss, d_vars)
                            g_tower_grads.append(g_grads)
                            d_tower_grads.append(d_grads)

            with tf.variable_scope('training'):
                # average the gradients back on the CPU
                with tf.variable_scope('average_gradients'):
                    with tf.variable_scope('generator'):
                        g_grads = average_gradients(g_tower_grads)
                    with tf.variable_scope('discriminator'):
                        d_grads = average_gradients(d_tower_grads)
                        
                # apply the gradients
                with tf.variable_scope('apply_gradients'):
                    with tf.variable_scope('generator'):
                        g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=global_step)
                    with tf.variable_scope('discriminator'):
                        d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=global_step)
                        
                # group training ops together
                with tf.variable_scope('train_op'):
                    self.train_op = tf.group(g_apply_gradient_op, d_apply_gradient_op)  

            # add summaries for the gradients
            for grad, var in g_grads + d_grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # collection all summary ops
            self.summary_op = tf.summary.merge(summaries)

    
    def construct_model(self, x, args, gpu_id):
        """Constructs the inference and loss portion of the model.
        Returns a list of form (latent, generator, real discriminator, 
        fake discriminator), where each is the final output node of each 
        component. """
        
        # latent
        with tf.variable_scope('latent'):
            z = self.build_latent(args.batch_size, args.latent_size)
            
        # generator
        with tf.variable_scope('generator'):
            g = self.build_generator(z, args.latent_size)
            
        # discriminator
        with tf.variable_scope('discriminator') as scope:
            # x = input, so restrict data to a slice for this particular gpu
            d_real = self.build_discriminator(x[gpu_id * args.batch_size:(gpu_id+1)*args.batch_size,:])
            scope.reuse_variables()
            d_fake = self.build_discriminator(g)
            
        # tf.summary.image('real_images', x, collections=['epoch'])
        # tf.summary.image('fake_images', g, collections=['epoch'])

        # losses
        with tf.variable_scope('losses'):
            # need to maximize this, but TF only does minimization, so we use negative
            with tf.variable_scope('generator'):
                g_loss = tf.reduce_mean(-tf.log(d_fake + 1e-8), name='g_loss')
            with tf.variable_scope('discriminator'):
                d_loss = tf.reduce_mean(-tf.log(d_real + 1e-8) - tf.log(1 - d_fake + 1e-8), name='d_loss')
        tf.add_to_collection('losses', g_loss)
        tf.add_to_collection('losses', d_loss)
        
        return (z, g, d_real, d_fake)
    
        
        
    def build_latent(self, batch_size, latent_size):
        z = tf.random_normal([batch_size, latent_size], 0, 1)                     ; self.activation_summary(z)
        tf.identity(z, name='sample')
        return z
    
    
    def build_generator(self, x, batch_size):
        """Output: 64x64x3."""
        # layer sizes = [96, 256, 256, 128, 64]
        with tf.variable_scope('fit_and_reshape'):
            x = tf.nn.relu(batch_norm(dense(x, batch_size, 32*4*4, name='d1')))
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        with tf.variable_scope('conv1'):
            x = tf.nn.relu(batch_norm(conv2d(x, 32, 96, 1, name='c1')))            ; self.activation_summary(x)
        with tf.variable_scope('conv2'):                
            x = tf.nn.relu(batch_norm(conv2d(x, 96, 256, 1, name='c2')))           ; self.activation_summary(x)
        with tf.variable_scope('deconv1'):                
            x = tf.nn.relu(batch_norm(deconv2d(x, 256, 256, 5, 2, name='dc1')))    ; self.activation_summary(x)
        with tf.variable_scope('deconv2'):                
            x = tf.nn.relu(batch_norm(deconv2d(x, 256, 128, 5, 2, name='dc2')))    ; self.activation_summary(x)
        with tf.variable_scope('deconv3'):                
            x = tf.nn.relu(batch_norm(deconv2d(x, 128, 64, 5, 2, name='dc3')))     ; self.activation_summary(x)
        with tf.variable_scope('deconv4'):
            x = tf.nn.relu(batch_norm(deconv2d(x, 64, 3, 5, 2, name='dc4')))       ; self.activation_summary(x)
        with tf.variable_scope('output'):
            x = tf.tanh(x, name='out')                                             ; self.activation_summary(x)
        tf.identity(x, name='sample')
        return x
    

    def build_discriminator(self, x):
        # layer sizes = [64, 128, 256, 256, 96]
        with tf.variable_scope('conv1'):
            x = lrelu(conv2d(x, 3, 64, 5, 2, name='c1'))                           ; self.activation_summary(x)
        with tf.variable_scope('conv2'):
            x = lrelu(batch_norm(conv2d(x, 64, 128, 5, 2, name='c2')))             ; self.activation_summary(x)
        with tf.variable_scope('conv3'):
            x = lrelu(batch_norm(conv2d(x, 128, 256, 5, 2, name='c3')))            ; self.activation_summary(x)
        with tf.variable_scope('conv4'):
            x = lrelu(batch_norm(conv2d(x, 256, 256, 5, 2, name='c4')))            ; self.activation_summary(x)
        with tf.variable_scope('conv5'):
            x = lrelu(batch_norm(conv2d(x, 256, 96, 32, 1, name='c5')))            ; self.activation_summary(x)
        with tf.variable_scope('conv6'):
            x = lrelu(batch_norm(conv2d(x, 96, 32, 1, name='c6')))                 ; self.activation_summary(x)
        with tf.variable_scope('logits'):
            logits = flatten(x, name='flat1')
            out = tf.nn.sigmoid(logits, name='sig1')                               ; self.activation_summary(out)
        tf.identity(out, name='sample')
        return out

    def tensor_name(self, x):
        return x.op.name
        # return re.sub('{}_[0-9]*/'.format('TOWER'), '', x.op.name)

    def activation_summary(self, x):
        n = self.tensor_name(x)
        tf.summary.histogram(n + '/activations', x)
        tf.summary.scalar(n + '/sparsity', tf.nn.zero_fraction(x))
        




    
