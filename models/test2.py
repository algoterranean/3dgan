import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from util import *
from models.model import Model
from models.ops import dense, conv2d, deconv2d, lrelu, flatten, montage_summary, input_slice



TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
SUMMARIES = tf.GraphKeys.SUMMARIES
UPDATE_OPS = tf.GraphKeys.UPDATE_OPS


class Test2(Model):
    def __init__(self, x, args):

        with tf.device('/cpu:0'):
            g_opt, d_opt = init_optimizer(args), init_optimizer(args)
            g_tower_grads, d_tower_grads = [], []
            global_step = tf.train.get_global_step()

            x = flatten(x)
            # rescale [0,1] to [-1,1]
            x = 2 * (x- 0.5)
            tf.summary.histogram('real', x)

        
            for scope, gpu_id in tower_scope_range(args.n_gpus):
                x_slice = input_slice(x, args.batch_size, gpu_id)
                g, g_scope = self.generator(args, reuse=(gpu_id>0))
                d_real, d_scope = self.discriminator(x_slice, args, reuse=(gpu_id>0))
                d_fake, d_scope = self.discriminator(g, args, reuse=True)

                g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                tf.get_variable_scope().reuse_variables()

                g_cost = -tf.reduce_mean(d_fake)
                d_cost = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)

                l_term = 10.0
                alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0.0, maxval=1.0)
                differences = g - x_slice
                interpolates = x_slice + (alpha * differences)
                d_interpolates, _ = self.discriminator(interpolates, args, reuse=True)
                gradients = tf.gradients(d_interpolates, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
                gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
                d_cost += l_term * gradient_penalty

                tf.add_to_collection('losses', g_cost)
                tf.add_to_collection('losses', d_cost)

                g_tower_grads.append(g_opt.compute_gradients(g_cost, var_list=g_params))
                d_tower_grads.append(d_opt.compute_gradients(d_cost, var_list=d_params))

                batchnorm_updates = tf.get_collection(UPDATE_OPS, scope)


                # rescale images from [-1,1] to [0,1]
                real_images = (x_slice[0:args.examples] + 1.0) / 2
                fake_samples = (g[0:args.examples] + 1.0) / 2
                # summaries
                montage_summary(tf.reshape(real_images, [-1, 64, 64, 3]), 8, 8, 'inputs')
                montage_summary(tf.reshape(fake_samples, [-1, 64, 64, 3]), 8, 8, 'fake')
                tf.summary.histogram('fakes', g)
                tf.summary.scalar('d_loss', d_cost)
                tf.summary.scalar('g_loss', g_cost)

                

            g_grads = average_gradients(g_tower_grads)
            d_grads = average_gradients(d_tower_grads)

            for grad, var in g_grads:
                tf.summary.histogram(var.name + '/gradient', grad)
            for grad, var in d_grads:
                tf.summary.histogram(var.name + '/gradient', grad)

            g_apply_gradient_op = g_opt.apply_gradients(g_grads, global_step=global_step)
            d_apply_gradient_op = d_opt.apply_gradients(d_grads, global_step=global_step)
            with tf.control_dependencies(batchnorm_updates):
                self.g_train_op = tf.group(g_apply_gradient_op)
            self.d_train_op = tf.group(d_apply_gradient_op)

                
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary_op = tf.summary.merge(summaries)


    def generator(self, args, reuse=False):
        output_dim = 64*64*3
        with tf.variable_scope('generator') as scope:
            z = tf.random_normal([args.batch_size, args.latent_size])
            y = tf.reshape(tf.nn.relu(batch_norm(dense(z, args.latent_size, 4*4*4*args.latent_size, name='fc1', reuse=reuse))), [-1, 4, 4, 4*args.latent_size])
            y = tf.nn.relu(batch_norm(deconv2d(y, 4*args.latent_size, 2*args.latent_size, 5, 2, name='dc1', reuse=reuse)))
            y = tf.nn.relu(batch_norm(deconv2d(y, 2*args.latent_size, args.latent_size, 5, 2, name='dc2', reuse=reuse)))
            y = tf.nn.relu(batch_norm(deconv2d(y, args.latent_size, args.latent_size, 5, 2, name='dc3', reuse=reuse)))
            y = tf.tanh(batch_norm(deconv2d(y, args.latent_size, 3, 5, 2, name='dc4', reuse=reuse)))
            y = tf.reshape(y, [-1, output_dim])
        return y, scope

    
    def discriminator(self, x, args, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            x = tf.reshape(x, [-1, 64, 64, 3])
            x = lrelu(conv2d(x, 3, args.latent_size, 5, 2, reuse=reuse, name='c1'))
            x = lrelu(conv2d(x, args.latent_size, args.latent_size*2, 5, 2, reuse=reuse, name='c2'))
            x = lrelu(conv2d(x, args.latent_size*2, args.latent_size*4, 5, 2, reuse=reuse, name='c3'))
            x = tf.reshape(x, [-1, 4*4*4*args.latent_size])
            x = dense(x, 4*4*4*args.latent_size, 1, reuse=reuse, name='fc2')
            x = tf.reshape(x, [-1])
        return x, scope


    def train(self, sess, args):
        for i in range(args.n_disc_train):
            sess.run(self.d_train_op)
        sess.run(self.g_train_op)

