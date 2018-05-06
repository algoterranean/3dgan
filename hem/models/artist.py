from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import math
import hem

class artist(hem.ModelPlugin):
    name = 'artist'

    @staticmethod
    def arguments():
        args = {
            '--examples': {
                'type': int,
                'default': 64,
                'help': 'Number of image summary examples to use.'}
            }
        return args


    @hem.default_to_cpu
    def __init__(self, x_y, args):
        x_opt = hem.init_optimizer(args)
        y_opt = hem.init_optimizer(args)
        x_decoder_tower_grads = []
        y_decoder_tower_grads = []
        global_step = tf.train.get_global_step()

        for x_y, scope, gpu_id in hem.tower_scope_range(x_y, args.n_gpus, args.batch_size):
            x = hem.rescale(x_y[0], (0, 1), (-1, 1))
            y = hem.rescale(x_y[1], (0, 1), (-1, 1))
            with tf.variable_scope('encoder'):
                e = artist.encoder(x, reuse=gpu_id>0)
            with tf.variable_scope('x_decoder'):
                x_hat = artist.decoder(e, args, channel_output=3, reuse=gpu_id>0)
            with tf.variable_scope('y_decoder'):
                y_hat = artist.decoder(e, args, channel_output=1, reuse=gpu_id>0)

            x_hat_loss, y_hat_loss = artist.losses(x, x_hat, y, y_hat, gpu_id==0)
            encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')
            x_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'x_decoder')
            y_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'y_decoder')
            # # train for x-reconstruction
            # x_decoder_tower_grads.append(x_opt.compute_gradients(x_hat_loss, var_list=encoder_vars + x_decoder_vars))
            # y_decoder_tower_grads.append(y_opt.compute_gradients(y_hat_loss, var_list=y_decoder_vars))
            # train for y-reconstruction
            x_decoder_tower_grads.append(x_opt.compute_gradients(x_hat_loss, var_list=x_decoder_vars))
            y_decoder_tower_grads.append(y_opt.compute_gradients(y_hat_loss, var_list=encoder_vars + y_decoder_vars))
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        x_grads = hem.average_gradients(x_decoder_tower_grads)
        y_grads = hem.average_gradients(y_decoder_tower_grads)
        with tf.control_dependencies(batchnorm_updates):
            self.x_train_op = x_opt.apply_gradients(x_grads, global_step=global_step)
            self.y_train_op = y_opt.apply_gradients(y_grads, global_step=global_step)
        self.x_hat = x_hat
        self.y_hat = y_hat
        self.x_hat_loss = x_hat_loss
        self.y_hat_loss = y_hat_loss
        self.all_losses = hem.collection_to_dict(tf.get_collection('losses'))

        artist.summaries(x, y, x_hat, y_hat, x_grads, y_grads, args)

    def train(self, sess, args, feed_dict):
        _, y_loss = sess.run([self.y_train_op, self.y_hat_loss], feed_dict=feed_dict)
        _, x_loss = sess.run([self.x_train_op, self.x_hat_loss], feed_dict=feed_dict)
        return {'x_loss': x_loss, 'y_loss': y_loss}

    @staticmethod
    def losses(x, x_hat, y, y_hat, add_to_collection=False):
        x = hem.rescale(x, (-1, 1), (0, 1))
        y = hem.rescale(y, (-1, 1), (0, 1))
        x_hat = hem.rescale(x_hat, (-1, 1), (0, 1))
        y_hat = hem.rescale(y_hat, (-1, 1), (0, 1))
        x_hat_loss = tf.reduce_mean(tf.square(x - x_hat), name='x_hat_loss')
        y_hat_loss = tf.reduce_mean(tf.square(y - y_hat), name='y_hat_loss')
        y_hat_rmse_loss = tf.sqrt(tf.reduce_mean(tf.square(y - y_hat)), name='y_hat_rmse')
        if add_to_collection:
            tf.add_to_collection('losses', x_hat_loss)
            tf.add_to_collection('losses', y_hat_loss)
            tf.add_to_collection('losses', y_hat_rmse_loss)
        return x_hat_loss, y_hat_loss

    @staticmethod
    def summaries(x, y, x_hat, y_hat, x_grads, y_grads, args):
        n = math.floor(math.sqrt(args.examples))
        with arg_scope([hem.montage],
                       height=n,
                       width=n):
            x = hem.rescale(x, (-1, 1), (0, 1))
            y = hem.rescale(y, (-1, 1), (0, 1))
            x_hat = hem.rescale(x_hat, (-1, 1), (0, 1))
            y_hat = hem.rescale(y_hat, (-1, 1), (0, 1))
            hem.montage(x[0:args.examples], name='x')
            hem.montage(y[0:args.examples], name='y', colorize=True)
            hem.montage(x_hat[0:args.examples], name='x_hat')
            hem.montage(y_hat[0:args.examples], name='y_hat', colorize=True)
        hem.summarize_gradients(x_grads)
        hem.summarize_gradients(y_grads)
        hem.summarize_losses()
        hem.summarize_activations(False)
        hem.summarize_weights_biases()

    @staticmethod
    def gradient_summaries(grads, name='gradients'):
        with tf.variable_scope(name):
            for g, v in grads:
                n = hem.tensor_name(v)
                tf.summary.histogram(n, g)
                tf.summary.scalar(n, tf.reduce_mean(g))

    @staticmethod
    def encoder(x, reuse=False):
        with arg_scope([hem.conv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       padding='VALID',
                       init=hem.xavier_initializer,
                       use_batch_norm=True,
                       activation=lambda x: hem.lrelu(x, leak=0.2)):
            x_rep = hem.conv2d(x,      3,  6, name='e1', use_batch_norm=False)
            x_rep = hem.conv2d(x_rep,  6, 12, name='e2')
            x_rep = hem.conv2d(x_rep, 12, 24, name='e3')
            x_rep = hem.conv2d(x_rep, 24, 48, name='e4')
            x_rep = hem.conv2d(x_rep, 48, 192, name='e5')
            x_rep = hem.conv2d(x_rep, 192, 384, name='e6')  # 5x5x192
        return x_rep

    @staticmethod
    def decoder(x_rep, args, channel_output=3, reuse=False):
        with arg_scope([hem.conv2d, hem.deconv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       padding='VALID',
                       init=hem.xavier_initializer,
                       use_batch_norm=True,
                       activation=lambda x: hem.lrelu(x, leak=0.2)):
            x_hat = hem.deconv2d(x_rep, 384, 192,
                                 output_shape=(args.batch_size, 192, 5, 5), name='d1')
            x_hat = hem.deconv2d(x_hat, 192, 48,
                                 output_shape=(args.batch_size, 48, 13, 13), name='d2')
            x_hat = hem.deconv2d(x_hat, 48, 24,
                                 output_shape=(args.batch_size, 24, 29, 29), name='d3')
            x_hat = hem.deconv2d(x_hat, 24, 12,
                                 output_shape=(args.batch_size, 12, 61, 61), name='d4')
            x_hat = hem.deconv2d(x_hat,  12, 6,
                                 output_shape=(args.batch_size, 6, 126, 126), name='d5')
            x_hat = hem.deconv2d(x_hat,  6, channel_output, activation=tf.tanh,
                                 output_shape=(args.batch_size, channel_output, 256, 256), name='d6')
        return x_hat

