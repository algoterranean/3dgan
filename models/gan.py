import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import time
import re
from sys import stdout
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, M
from util import print_progress, fold, average_gradients, init_optimizer

from tensorflow import variable_scope as var_scope, name_scope as name_scope
# from tensorflow.nn import relu
relu = tf.nn.relu

# batch norm info:
# https://github.com/tensorflow/tensorflow/issues/4361

class GAN(Model):
    """Vanilla Generative Adversarial Network with multi-GPU support.

    All variables are pinned to the CPU and shared between GPUs. 
    Gradients are collected on the CPU, averaged, and applied back to each GPU.
    We only keep one set of batch norm updates and include it as part of the train op.
    """
    
    def __init__(self, x, global_step, args):
        with tf.device('/cpu:0'):
            # store optimizer on CPU
            
            with var_scope('optimizers'):
                with var_scope('generator'):
                    g_opt = init_optimizer(args)
                with var_scope('disciminator'):
                    d_opt = init_optimizer(args)

            g_tower_grads = []
            d_tower_grads = []

            # build model on each GPU
            with var_scope('model'): # tf.get_variable_scope()):
                for gpu_id in range(args.n_gpus):
                    with tf.device(self.variables_on_cpu(gpu_id)):
                        with name_scope('tower_{}'.format(gpu_id)) as scope:
                            # build model
                            z, g, d_real, d_fake = self.construct_model(x, args, gpu_id)

                            # generator summaries
                            with var_scope('generator_summaries'):
                                s = tf.expand_dims(tf.concat(tf.unstack(g, num=args.batch_size, axis=0)[0:10], axis=1), axis=0)
                                # self.fake_image_name = tf.placeholder(tf.string, [])
                                # tf.summary.image("fake_" + tf.cast(global_step, tf.string), s)
                                # tf.string_join(['fake_images_', tf.cast(global_step, tf.string)])
                                tf.summary.image('fake_images', s)
                                
                            # get losses and add summaries
                            losses = tf.get_collection('losses', scope)
                            for l in losses:
                                tf.summary.scalar(self.tensor_name(l), l)
                            g_loss, d_loss = losses

                            # reuse variables in next tower
                            tf.get_variable_scope().reuse_variables()

                            # add summaries for generated images and real images
                            # tf.summary.image('real_images', x)
                            # tf.summary.image('fake_images', g)

                            # use the last tower's summaries
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # keep only one batchnorm update since one accumulates rapidly enough for both
                            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                            # g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/generator')
                            # d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator')

                            # compute and collect gradients for generator and discriminator
                            # restrict optimizer to vars for each component
                            params = tf.trainable_variables()
                            g_vars = [i for i in params if 'generator' in i.name]
                            d_vars = [i for i in params if 'discriminator' in i.name]

                            with var_scope('compute_gradients'):
                                with var_scope('generator'):
                                    g_grads = g_opt.compute_gradients(g_loss, var_list=g_vars)
                                with var_scope('discriminator'):
                                    d_grads = d_opt.compute_gradients(d_loss, var_list=d_vars)

                            g_tower_grads.append(g_grads)
                            d_tower_grads.append(d_grads)

                            
            with var_scope('training'):
                # average the gradients back on the CPU
                with var_scope('average_gradients'):
                    with var_scope('generator'):  
                        g_grads = average_gradients(g_tower_grads)
                    with var_scope('discriminator'):
                        d_grads = average_gradients(d_tower_grads)
                        
                # apply the gradients
                with var_scope('apply_gradients'):
                    with var_scope('generator'):
                        g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=global_step)
                    with var_scope('discriminator'):
                        d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=global_step)
                        
                # group training ops together
                with var_scope('train_op'):
                    batchnorm_updates_op = tf.group(*batchnorm_updates)
                    self.train_op = tf.group(g_apply_gradient_op, d_apply_gradient_op, batchnorm_updates_op)  

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
        with var_scope('latent'):
            z = self.build_latent(args.batch_size, args.latent_size)
        # generator
        with var_scope('generator'):
            g = self.build_generator(z, args.latent_size, (gpu_id > 0))
        # discriminator
        with var_scope('discriminator') as dscope:
            # split batch between GPUs
            with var_scope('sliced_input'):
                sliced_input = x[gpu_id * args.batch_size:(gpu_id+1)*args.batch_size,:]
            # d_real = self.build_discriminator(x, (gpu_id>0))
            d_real = self.build_discriminator(sliced_input, (gpu_id>0))
            d_fake = self.build_discriminator(g, True)
            
        # tf.summary.image('real_images', x, collections=['epoch'])
        # tf.summary.image('fake_images', g, collections=['epoch'])

        # losses
        with var_scope('losses'):
            # need to maximize this, but TF only does minimization, so we use negative
            with var_scope('generator'):
                g_loss = tf.reduce_mean(-tf.log(d_fake + 1e-8), name='g_loss')
            with var_scope('discriminator'):
                d_loss = tf.reduce_mean(-tf.log(d_real + 1e-8) - tf.log(1 - d_fake + 1e-8), name='d_loss')
        tf.add_to_collection('losses', g_loss)
        tf.add_to_collection('losses', d_loss)
        
        return (z, g, d_real, d_fake)
    
            
    def build_latent(self, batch_size, latent_size):
        z = tf.random_normal([batch_size, latent_size], 0, 1)                     ; self.activation_summary(z)
        tf.identity(z, name='sample')
        return z

    
    def variables_on_cpu(self, gpu_id):
        def helper(op):
            return '/cpu:0' if op.type == 'VariableV2' else '/gpu:{}'.format(gpu_id)
        return helper

    
    def build_generator(self, x, batch_size, reuse): #, device_name='/gpu:0'):
        """Output: 64x64x3."""
        # layer sizes = [96, 256, 256, 128, 64]
        with var_scope('fit_and_reshape', reuse=reuse):
            x = relu(batch_norm(dense(x, batch_size, 32*4*4, reuse, name='d1')))
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        with var_scope('conv1', reuse=reuse):
            b = conv2d(x, 32, 96, 1, reuse=reuse, name='c1')
            x = relu(batch_norm(b))            ; self.activation_summary(x)
        with var_scope('conv2', reuse=reuse):                
            x = relu(batch_norm(conv2d(x, 96, 256, 1, reuse=reuse, name='c2')))           ; self.activation_summary(x)
        with var_scope('deconv1', reuse=reuse):                
            x = relu(batch_norm(deconv2d(x, 256, 256, 5, 2, reuse=reuse, name='dc1')))    ; self.activation_summary(x)
        with var_scope('deconv2', reuse=reuse):                
            x = relu(batch_norm(deconv2d(x, 256, 128, 5, 2, reuse=reuse, name='dc2')))    ; self.activation_summary(x)
        with var_scope('deconv3', reuse=reuse):                
            x = relu(batch_norm(deconv2d(x, 128, 64, 5, 2, reuse=reuse, name='dc3')))     ; self.activation_summary(x)
        with var_scope('deconv4', reuse=reuse):
            x = relu(batch_norm(deconv2d(x, 64, 3, 5, 2, reuse=reuse, name='dc4')))       ; self.activation_summary(x)
        with var_scope('output'):
            x = tf.tanh(x, name='out')                                                          ; self.activation_summary(x)
        tf.identity(x, name='sample')
        return x
    

    def build_discriminator(self, x, reuse):
        # layer sizes = [64, 128, 256, 256, 96]
        with var_scope('conv1', reuse=reuse):
            x = lrelu(conv2d(x, 3, 64, 5, 2, reuse=reuse, name='c1'))                           ; self.activation_summary(x)
        with var_scope('conv2', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 64, 128, 5, 2, reuse=reuse, name='c2')))             ; self.activation_summary(x)
        with var_scope('conv3', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 128, 256, 5, 2, reuse=reuse, name='c3')))            ; self.activation_summary(x)
        with var_scope('conv4', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 256, 256, 5, 2, reuse=reuse, name='c4')))            ; self.activation_summary(x)
        with var_scope('conv5', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 256, 96, 32, 1, reuse=reuse, name='c5')))            ; self.activation_summary(x)
        with var_scope('conv6', reuse=reuse):
            x = lrelu(batch_norm(conv2d(x, 96, 32, 1, reuse=reuse, name='c6')))                 ; self.activation_summary(x)
        with var_scope('logits'):
            logits = flatten(x, name='flat1')
            out = tf.nn.sigmoid(logits, name='sig1')                                            ; self.activation_summary(out)
        tf.identity(out, name='sample')
        return out

    def tensor_name(self, x):
        # return x.name
        return re.sub('{}_[0-9]*/'.format('tower'), '', x.op.name)

    def activation_summary(self, x):
        n = self.tensor_name(x)
        tf.summary.histogram(n + '/activations', x)
        tf.summary.scalar(n + '/sparsity', tf.nn.zero_fraction(x))
        

