from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import math
import hem

class info_gan(hem.ModelPlugin):
    name = 'info_gan'

    @hem.default_to_cpu
    def __init__(self, x_y, args):
        g_opt = hem.init_optimizer(args)
        d_opt = hem.init_optimizer(args)
        q_opt = hem.init_optimizer(args)

        x = hem.rescale(x_y[0], (0, 1), (-1, 1)) # 256x256x3
        y = hem.rescale(x_y[1], (0, 1), (-1, 1)) # 256x256x1
        z = tf.random_uniform((args.batch_size, 1, 256, 256)) # 256x256x1
        with tf.variable_scope('generator') as scope:
            g = info_gan.generator(z, x)
        with tf.variable_scope('discriminator') as scope:
            d_real = info_gan.discriminator(y)
            d_fake = info_gan.discriminator(g, reuse=True)
        with tf.variable_scope('predictor') as scope:
            q = info_gan.predictor(g)

        g_loss = -tf.reduce_mean(tf.log(d_fake + 1e-8))
        d_loss = -tf.reduce_mean(tf.log(d_real + 1e-8) + tf.log(1 - d_fake + 1e-8))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(q + 1e-8) * x), axis=1)
        entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(x + 1e-8) * x), axis=1)
        q_loss = cross_entropy + entropy
        for l in [g_loss, d_loss, q_loss]:
            tf.add_to_collection('losses', l)
        self.all_losses = hem.collection_to_dict(tf.get_collection('losses'))

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'predictor')

        self.g_train_op = g_opt.minimize(g_loss, var_list=g_vars)
        self.d_train_op = d_opt.minimize(d_loss, var_list=d_vars)
        self.q_train_op = q_opt.minimize(q_loss, var_list=q_vars + g_vars)

    def train(self, sess, args, feed_dict):
        d_loss = sess.run(self.d_train_op, feed_dict=feed_dict)
        g_loss = sess.run(self.g_train_op, feed_dict=feed_dict)
        q_loss = sess.run(self.q_train_op, feed_dict=feed_dict)
        return {'d_loss': d_loss, 'g_loss': g_loss, 'q_loss': q_loss}


    @staticmethod
    def generator(z, x, reuse=False):
        # given the noise z (256x256x1) and rgb map x (256x256x3), generate a depth map y (256x256x1)
        with tf.variable_scope('generator', reuse=reuse),\
             arg_scope([hem.conv2d, hem.deconv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       padding='VALID',
                       init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                       activation=lambda x: hem.lrelu(x, leak=0.2)):
            y = tf.concat([x, z], axis=1)
            y = hem.conv2d(y,   4,  64, name='g1')
            y = hem.conv2d(y,  64, 128, name='g2')
            y = hem.conv2d(y, 128, 256, name='g3')
            y = hem.conv2d(y, 256, 512, name='g4')
            y = hem.deconv2d(y, 512, 256, name='g5')
            y = hem.deconv2d(y, 256, 128, name='g6')
            y = hem.deconv2d(y, 128,  64, name='g7')
            y = hem.deconv2d(y,  64,   1, name='g8', activation=tf.tanh)
        return y



    @staticmethod
    def discriminator(y, reuse=False):
        # given a depth map y, determine whether it is real or fake
        with tf.variable_scope('discriminator', reuse=reuse),\
             arg_scope([hem.conv2d],
                 reuse=reuse,
                 filter_size=5,
                 stride=2,
                 padding='VALID',
                 init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                 activation=lambda x: hem.lrelu(x, leak=0.2)):
            y = hem.conv2d(y,   1,  64, name='d1')
            y = hem.conv2d(y,  64, 128, name='d2')
            y = hem.conv2d(y, 128, 256, name='d3')
            y = hem.conv2d(y, 256, 512, name='d4')
            y = hem.conv2d(y, 512, 256, name='d5')
            y = hem.conv2d(y, 256,   1, name='d6', activation=tf.sigmoid)
        return y


    @staticmethod
    def predictor(y, reuse=False):
        # given the depth map y, generate a rgb scene x
        with tf.variable_scope('predictor, reuse=reuse'),\
             arg_scope([hem.conv2d, hem.deconv2d],
                 reuse=reuse,
                 filter_size=5,
                 stride=2,
                 padding='VALID',
                 init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                 activation=lambda x: hem.lrelu(x, leak=0.2)):
             y = hem.conv2d(y, 1, 3, stride=1, activation=tf.tanh)
        return y
