"""Implementation of traditional convolutional autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from util import average_gradients, init_optimizer
from model import Model
from ops.layers import dense, conv2d, deconv2d, flatten
from ops.summaries import montage_summary, activation_summary
from ops.input import input_slice
from ops.activations import lrelu


# TODO: build decoder from same weights as encoder

class CNN(Model):
    def __init__(self, x, args):
        with tf.device('/cpu:0'):
            opt = init_optimizer(args)
            tower_grads = []

            with tf.variable_scope('model') as scope:
                for gpu_id in range(args.n_gpus):
                    with tf.device(self.variables_on_cpu(gpu_id)):
                        with tf.name_scope('tower_{}'.format(gpu_id)) as scope:
                            # input
                            x_slice = input_slice(x, args.batch_size, gpu_id)
                            # model
                            encoder, latent, decoder = self.build_model(x_slice, args, gpu_id)
                            # loss
                            loss = self.summarize_collection('losses', scope)[0]
                            # reuse variables on next tower
                            tf.get_variable_scope().reuse_variables()
                            # add montage summaries
                            montage_summary(x_slice, 8, 8, 'inputs')
                            montage_summary(decoder, 8, 8, 'outputs')
                            # compute gradients on this GPU
                            with tf.variable_scope('compute_gradients'):
                                tower_grads.append(opt.compute_gradients(loss))
            # back on the CPU
            with tf.variable_scope('training'):
                avg_grads, apply_grads, self.train_op = self.average_and_apply_grads(tower_grads, opt)
                self.summarize_grads(avg_grads)
                
            # combine summaries
            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))


    def build_encoder(self, x, reuse=False):
        """Input: 64x64x3. Output: 4x4x32.
        Layer sizes = 64, 128, 256, 256, 96, 32"""
        
        with tf.variable_scope('conv1'):
            x = lrelu(conv2d(x, 3, 64, 5, 2, reuse=reuse, name='c1'))
            activation_summary(x, 8, 8)
        with tf.variable_scope('conv2'):
            x = lrelu(conv2d(x, 64, 128, 5, 2, reuse=reuse, name='c2'))
            activation_summary(x, 8, 16)
        with tf.variable_scope('conv3'):
            x = lrelu(conv2d(x, 128, 256, 5, 2, reuse=reuse, name='c3'))
            activation_summary(x, 16, 16)
        with tf.variable_scope('conv4'):
            x = lrelu(conv2d(x, 256, 256, 5, 2, reuse=reuse, name='c4'))
            activation_summary(x, 16, 16)
        with tf.variable_scope('conv5'):
            x = lrelu(conv2d(x, 256, 96, 1, reuse=reuse, name='c5'))
            activation_summary(x, 12, 8)
        with tf.variable_scope('conv6'):
            x = lrelu(conv2d(x, 96, 32, 1, reuse=reuse, name='c6'))
            activation_summary(x, 8, 4)
        tf.identity(x, name='sample')
        return x

    def build_decoder(self, x, latent_size, reuse=False):
        """Input: 200. Output: 64x64x3.
        Layer sizes = 96, 256, 256, 128, 64"""

        with tf.variable_scope('dense'):
            x = dense(x, latent_size, 32*4*4, reuse=reuse, name='d1')
            activation_summary(x, montage=False)
        with tf.variable_scope('conv1'):
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
            x = tf.nn.relu(conv2d(x, 32, 96, 1, reuse=reuse, name='c1'))
            activation_summary(x, 8, 4)
        with tf.variable_scope('conv2'):
            x = tf.nn.relu(conv2d(x, 96, 256, 1, reuse=reuse, name='c2'))
            activation_summary(x, 16, 16)
        with tf.variable_scope('deconv1'):
            x = tf.nn.relu(deconv2d(x, 256, 256, 5, 2, reuse=reuse, name='dc1'))
            activation_summary(x, 16, 16)
        with tf.variable_scope('deconv2'):
            x = tf.nn.relu(deconv2d(x, 256, 128, 5, 2, reuse=reuse, name='dc2'))
            activation_summary(x, 8, 16)
        with tf.variable_scope('deconv3'):
            x = tf.nn.relu(deconv2d(x, 128, 64, 5, 2, reuse=reuse, name='dc3'))
            activation_summary(x, 8, 8)
        with tf.variable_scope('deconv4'):
            x = tf.nn.sigmoid(deconv2d(x, 64, 3, 5, 2, reuse=reuse, name='dc4'))
            activation_summary(x, montage=False)
        tf.identity(x, name='sample')
        return x


    def build_latent(self, x, latent_size, reuse=False):
        with tf.variable_scope('flatten'):
            flat = flatten(x)
        with tf.variable_scope('dense'):
            x = dense(flat, 32*4*4, latent_size, reuse=reuse, name='d1')
            self.activation_summary(x)
        return x
    

    def build_model(self, x, args, gpu_id):
        with tf.variable_scope('encoder'):
            encoder = self.build_encoder(x, (gpu_id > 0))
        with tf.variable_scope('latent'):
            latent = self.build_latent(encoder, args.latent_size, (gpu_id > 0))
        with tf.variable_scope('decoder'):
            decoder = self.build_decoder(latent, args.latent_size, (gpu_id > 0))
        with tf.variable_scope('losses'):
            loss = tf.reduce_mean(tf.abs(x - decoder), name='loss')
        tf.add_to_collection('losses', loss)
        return (encoder, latent, decoder)
