import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import time
from sys import stdout
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, M
from util import print_progress, fold, average_gradients, init_optimizer




class GAN(Model):
    """Vanilla Generative Adversarial Network with multi-GPU support."""
    def __init__(self, x, global_step, args): #batch_size, latent_size, g_opt, d_opt):
        with tf.device('/cpu:0'):
            # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            g_opt = init_optimizer(args)
            d_opt = init_optimizer(args)

            g_tower_grads = []
            d_tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_id in range(args.n_gpus):
                    with tf.device('/gpu:{}'.format(gpu_id)):
                        with tf.name_scope('TOWER_{}'.format(gpu_id)) as scope:
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
                            tf.summary.image('real_images', x, collections=['epoch'])
                            tf.summary.image('fake_images', g, collections=['epoch'])
            
                            # losses
                            with tf.variable_scope('losses'):
                                # need to maximize this, but TF only does minimization, so we use negative
                                with tf.variable_scope('generator'):
                                    g_loss = tf.reduce_mean(-tf.log(d_fake + 1e-8), name='g_loss')
                                with tf.variable_scope('discriminator'):
                                    d_loss = tf.reduce_mean(-tf.log(d_real + 1e-8) - tf.log(1 - d_fake + 1e-8), name='d_loss')
                            tf.add_to_collection('losses', g_loss)
                            tf.add_to_collection('losses', d_loss)
                            tf.summary.scalar('loss_g', g_loss)
                            tf.summary.scalar('loss_d', d_loss)
    
                            g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/generator')
                            d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator')
    
                            tf.get_variable_scope().reuse_variables()
                            with tf.variable_scope('gradients'):
                                with tf.variable_scope('generator'):
                                    g_grads  = g_opt.compute_gradients(g_loss, g_vars)
                                with tf.variable_scope('discriminator'):
                                    d_grads = d_opt.compute_gradients(d_loss, d_vars)
                            g_tower_grads.append(g_grads)
                            d_tower_grads.append(d_grads)
    
            
            # back on CPU
            with tf.variable_scope('average_gradients'):
                with tf.variable_scope('generator'):
                    g_grads = average_gradients(g_tower_grads)
                with tf.variable_scope('discriminator'):
                    d_grads = average_gradients(d_tower_grads)
            with tf.variable_scope('apply_gradients'):
                with tf.variable_scope('generator'):
                    g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=global_step)
                with tf.variable_scope('discriminator'):
                    d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=global_step)
    
            self.train_op = tf.group(g_apply_gradient_op, d_apply_gradient_op)
            
        
        
    def build_latent(self, batch_size, latent_size):
        z = tf.random_normal([batch_size, latent_size], 0, 1)
        tf.identity(z, name='sample')
        return z
    
    
    def build_generator(self, x, batch_size):
        """Output: 64x64x3."""
        # layer sizes = [96, 256, 256, 128, 64]
        with tf.variable_scope('fit_and_reshape'):
        # with tf.variable_scope('fit_and_reshape'):
            x = tf.nn.relu(batch_norm(dense(x, batch_size, 32*4*4, name='d1')))
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
        with tf.variable_scope('conv1'):
            x = tf.nn.relu(batch_norm(conv2d(x, 32, 96, 1, name='c1')))            ; M(x, 'layer')
        with tf.variable_scope('conv2'):                
            x = tf.nn.relu(batch_norm(conv2d(x, 96, 256, 1, name='c2')))           ; M(x, 'layer')
        with tf.variable_scope('deconv1'):                
            x = tf.nn.relu(batch_norm(deconv2d(x, 256, 256, 5, 2, name='dc1')))     ; M(x, 'layer')
        with tf.variable_scope('deconv2'):                
            x = tf.nn.relu(batch_norm(deconv2d(x, 256, 128, 5, 2, name='dc2')))     ; M(x, 'layer')
        with tf.variable_scope('deconv3'):                
            x = tf.nn.relu(batch_norm(deconv2d(x, 128, 64, 5, 2, name='dc3')))     ; M(x, 'layer')
        with tf.variable_scope('deconv4'):
            x = tf.nn.relu(batch_norm(deconv2d(x, 64, 3, 5, 2, name='dc4')))       ; M(x, 'layer')
        with tf.variable_scope('output'):
            x = tf.tanh(x, name='out')                                             ; M(x, 'layer')
        tf.identity(x, name='sample')
        return x
    

    def build_discriminator(self, x):
        # layer sizes = [64, 128, 256, 256, 96]
        with tf.variable_scope('conv1'):
            x = lrelu(conv2d(x, 3, 64, 5, 2, name='c1'))                           ; M(x, 'layer')
        with tf.variable_scope('conv2'):
            x = lrelu(batch_norm(conv2d(x, 64, 128, 5, 2, name='c2')))             ; M(x, 'layer')
        with tf.variable_scope('conv3'):
            x = lrelu(batch_norm(conv2d(x, 128, 256, 5, 2, name='c3')))            ; M(x, 'layer')
        with tf.variable_scope('conv4'):
            x = lrelu(batch_norm(conv2d(x, 256, 256, 5, 2, name='c4')))            ; M(x, 'layer')
        with tf.variable_scope('conv5'):
            x = lrelu(batch_norm(conv2d(x, 256, 96, 32, 1, name='c5')))            ; M(x, 'layer')
        with tf.variable_scope('conv6'):
            x = lrelu(batch_norm(conv2d(x, 96, 32, 1, name='c6')))                 ; M(x, 'layer')
        with tf.variable_scope('logits'):
            logits = flatten(x, name='flat1')            
            out = tf.nn.sigmoid(logits, name='sig1')                               ; M(out, 'layer')
        tf.identity(out, name='sample')
        return out




    
