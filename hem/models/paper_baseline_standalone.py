from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import math
import hem


# TODO Add summaries for all GPUs

class paper_baseline_standalone(hem.ModelPlugin):
    name = 'paper_baseline_standalone'

    @staticmethod
    def arguments():
        args = {
            '--g_lr': {
                'type': float,
                'default': 1e-3,
                'help': 'Learning rate for generator.' },
            '--g_beta1': {
                'type': float,
                'default': 0.9,
                'help': 'Beta1 for generator' },
            '--g_beta2': {
                'type': float,
                'default': 0.999,
                'help': 'Beta2 for generator.' },
            '--model_version': {
                'type':str,
                'default': 'baseline',
                'choices': ['baseline', 'mean_adjusted', 'mean_provided'], #, 'mean_scene_provided'],
                'help': 'Which version of the model to run."' }
            }
        return args

    @hem.default_to_cpu
    def __init__(self, x_y, args):
        # init/setup
        g_opt = tf.train.AdamOptimizer(args.g_lr, args.g_beta1, args.g_beta2)
        g_tower_grads = []
        global_step = tf.train.get_global_step()

        self.mean_image_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 29, 29))
        # self.var_image_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 29, 29))

        # foreach gpu...
        for x_y, scope, gpu_id in hem.tower_scope_range(x_y, args.n_gpus, args.batch_size):
            with tf.variable_scope('input_preprocess'):
                # split inputs and rescale
                x = x_y[0]
                y = x_y[1]
                # re-attach shape info
                x = tf.reshape(x, (args.batch_size, 3, 65, 65))
                # rescale from [0,1] to actual world depth
                y = y * 10.0
                y = hem.crop_to_bounding_box(y, 17, 17, 29, 29)
                # re-attach shape info
                y = tf.reshape(y, (args.batch_size, 1, 29, 29))
                y_bar = tf.reduce_mean(y, axis=[2, 3], keep_dims=True)



            # create model
            with tf.variable_scope('generator'):
                if args.model_version == 'baseline':
                    g = self.g_baseline(x, args, reuse=(gpu_id > 0))
                    g_0 = tf.zeros_like(g)
                    y_hat = g
                    y_0 = g_0
                elif args.model_version == 'mean_adjusted':
                    g = self.g_baseline(x, args, reuse=(gpu_id > 0))
                    g_0 = tf.zeros_like(g)
                    y_hat = g + y_bar
                    y_0 = g_0 + y_bar
                elif args.model_version == 'mean_provided':
                    g = self.g_mean_provided(x, y_bar, args, reuse=(gpu_id > 0))
                    g_0 = tf.zeros_like(g)
                    y_hat = g + y_bar
                    y_0 = g_0 + y_bar

                # if gpu_id == 0:
                #     tf.summary.histogram('g', g)
                #     tf.summary.histogram('y_hat', y_hat)
                #     tf.summary.histogram('y_0', y_0)
                #     hem.montage(g, num_examples=64, width=8, height=8, name='g')
                #     hem.montage(y_hat, num_examples=64, width=8, height=8, name='y_hat')
                #     hem.montage(y_0, num_examples=64, width=8, height=8, name='y_0')

            # calculate losses
            g_loss = self.loss(x, y, y_hat, args, reuse=(gpu_id > 0))
            # calculate gradients
            g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
            g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_params))

        # average and apply gradients
        g_grads = hem.average_gradients(g_tower_grads, check_numerics=args.check_numerics)
        g_apply_grads = g_opt.apply_gradients(g_grads, global_step=global_step)

        # add summaries
        hem.summarize_losses()
        hem.summarize_gradients(g_grads, name='g_gradients')
        generator_layers = [l for l in tf.get_collection('conv_layers') if 'generator' in l.name]
        hem.summarize_layers('g_activations', generator_layers, montage=True)
        self.montage_summaries(x, y, g, y_hat, args)
        self.metric_summaries(x, y, g, y_hat, args, name='y_hat')
        self.metric_summaries(x, y, g_0, y_0, args, name='y_0')
        self.metric_summaries(x, y, g, self.mean_image_placeholder * 10.0, args, name='y_mean')

        # training ops
        self.g_train_op = g_apply_grads
        self.all_losses = hem.collection_to_dict(tf.get_collection('losses'))


    def train(self, sess, args, feed_dict):
        _, results = sess.run([self.g_train_op, self.all_losses], feed_dict=feed_dict)
        return results

    def g_baseline(self, x, args, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse), \
             arg_scope([hem.conv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       padding='VALID',
                       # init=lambda: tf.truncated_normal_initializer(mean=0, stddev=0.01),
                       init=tf.contrib.layers.xavier_initializer,
                       activation=tf.nn.relu):        # 65x65x3
            e1 = hem.conv2d(x,    3,  64, name='e1')  # 31x31x64
            e2 = hem.conv2d(e1,  64, 128, name='e2')  # 14x14x128
            e3 = hem.conv2d(e2, 128, 256, name='e3')  # 5x5x256
            e4 = hem.conv2d(e3, 256, 512, name='e4')  # 1x1x512
        with tf.variable_scope('decoder', reuse=reuse), \
             arg_scope([hem.deconv2d, hem.conv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       # init=tf.truncated_normal_initializer,
                       # init=lambda: tf.truncated_normal_initializer(mean=0, stddev=0.01),
                       init=tf.contrib.layers.xavier_initializer,
                       padding='VALID',
                       activation=lambda x: hem.lrelu(x, leak=0.2)):                                                # 1x1x512
            y_hat = hem.deconv2d(e4, 512, 256, output_shape=(args.batch_size, 256, 5, 5), name='d1')                # 5x5x256
            y_hat = tf.concat([y_hat, e3], axis=1)                                                                  # 5x5x512
            y_hat = hem.deconv2d(y_hat, 512, 128, output_shape=(args.batch_size, 128, 14, 14), name='d2')           # 14x14x128
            y_hat = tf.concat([y_hat, e2], axis=1)                                                                  # 14x14x256
            y_hat = hem.deconv2d(y_hat, 256, 64, output_shape=(args.batch_size, 64, 31, 31), name='d3')             # 31x31x64
            y_hat = tf.concat([y_hat, e1], axis=1)                                                                  # 31x31x128
            y_hat = hem.conv2d(y_hat, 128, 1, stride=1, filter_size=1, padding='SAME', activation=None, name='d4')  # 31x31x1
            y_hat = hem.crop_to_bounding_box(y_hat, 0, 0, 29, 29)                                                   # 29x29x1
            # y_hat = tf.maximum(y_hat, tf.zeros_like(y_hat))
        return y_hat


    def g_mean_provided(self, x, y_bar, args, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse), \
             arg_scope([hem.conv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       padding='VALID',
                       init=tf.contrib.layers.xavier_initializer,
                       activation=tf.nn.relu):        # 65x65x3
            e1 = hem.conv2d(x,    3,  64, name='e1')  # 31x31x64
            e1 = tf.concat([e1, tf.ones((args.batch_size, 1, 31, 31)) * y_bar], axis=1)
            e2 = hem.conv2d(e1,  65, 128, name='e2')  # 14x14x128
            e3 = hem.conv2d(e2, 128, 256, name='e3')  # 5x5x256
            e4 = hem.conv2d(e3, 256, 512, name='e4')  # 1x1x512
        with tf.variable_scope('decoder', reuse=reuse), \
             arg_scope([hem.deconv2d, hem.conv2d],
                       reuse=reuse,
                       filter_size=5,
                       stride=2,
                       init=tf.contrib.layers.xavier_initializer,
                       padding='VALID',
                       activation=lambda x: hem.lrelu(x, leak=0.2)):                                                # 1x1x512
            y_hat = hem.deconv2d(e4, 512, 256, output_shape=(args.batch_size, 256, 5, 5), name='d1')                # 5x5x256
            y_hat = tf.concat([y_hat, e3], axis=1)                                                                  # 5x5x512
            y_hat = hem.deconv2d(y_hat, 512, 128, output_shape=(args.batch_size, 128, 14, 14), name='d2')           # 14x14x128
            y_hat = tf.concat([y_hat, e2], axis=1)                                                                  # 14x14x256
            y_hat = hem.deconv2d(y_hat, 256, 64, output_shape=(args.batch_size, 64, 31, 31), name='d3')             # 31x31x64
            y_hat = tf.concat([y_hat, e1], axis=1)                                                                  # 31x31x128
            y_hat = hem.conv2d(y_hat, 129, 1, stride=1, filter_size=1, padding='SAME', activation=None, name='d4')  # 31x31x1
            y_hat = hem.crop_to_bounding_box(y_hat, 0, 0, 29, 29)                                                   # 29x29x1
            # y_hat = tf.maximum(y_hat, tf.zeros_like(y_hat))
        return y_hat

    def loss(self, x, y, y_hat, args, reuse=False):
        with tf.variable_scope('loss'):
            y_bar = tf.reduce_mean(y, axis=[2, 3], keep_dims=True)
            y = y / 10.0
            y_hat = y_hat / 10.0
            y_bar = y_bar / 10.0
            l = hem.rmse(y, y_hat)
            if not reuse:
                hem.add_to_collection('losses', [l])
        return l

    # summaries
    ############################################
    def montage_summaries(self, x, y, g, y_hat, args):
        n_examples = 64
        n = math.floor(math.sqrt(n_examples))
        with tf.variable_scope('montage_preprocess'):
            y_bar = tf.reduce_mean(y, axis=[2, 3], keep_dims=True)
            y_hat = tf.reshape(y_hat, tf.shape(y))  # reattach shape info
            g = tf.reshape(g, tf.shape(y))  # reattach shape info
            y = y / 10.0
            y_hat = y_hat / 10.0
            y_bar = y_bar / 10.0
            g = g / 10.0

        with arg_scope([hem.montage],
                       num_examples=n_examples,
                       height=n,
                       width=n,
                       colorize=True):
            with tf.variable_scope('model'):
                hem.montage(x,     name='x')
                hem.montage(y,     name='y')
                hem.montage(tf.ones_like(y) * y_bar, name='y_bar')
                hem.montage(g,     name='g')
                hem.montage(y_hat, name='y_hat')


    def metric_summaries(self, x, y, g, y_hat, args, name=None):
        # from Eigen et. al 2014
        ns = 'metrics' if name is None else 'metrics_' + name
        with tf.variable_scope(ns):
            g = g / 10.0
            y = y / 10.0
            y_hat = y_hat / 10.0

            # standard pixel-wise difference metrics
            abs_rel_diff = tf.reduce_mean(tf.abs(y - y_hat)/y_hat, name='abs_rel_diff')
            squared_rel_diff = tf.reduce_mean(tf.square(y - y_hat)/y_hat)
            linear_rmse = hem.rmse(y, y_hat, name='linear_rmse')
            log_rmse = hem.rmse(tf.log(y + 1e-8), tf.log(y_hat + 1e-8), name='log_rmse')
            tf.summary.scalar('abs_rel_diff', abs_rel_diff)
            tf.summary.scalar('squared_rel_diff', squared_rel_diff)
            tf.summary.scalar('linear_rmse', linear_rmse)
            tf.summary.scalar('log_rmse', log_rmse)

            # scale-invariant rmse
            d = tf.log(y + 1e-8) - tf.log(y_hat + 1e-8)
            n = tf.cast(tf.size(d), tf.float32) # tf.size() = 430592
            scale_invariant_log_rmse = tf.reduce_mean(tf.square(d)) - (tf.reduce_sum(d) ** 2)/(n**2)
            tf.summary.scalar('scale_invariant_log_rmse', scale_invariant_log_rmse)

            # threshold metrics
            delta = tf.maximum(y/y_hat, y_hat/y)
            t1, t1_op = tf.metrics.percentage_below(delta, 1.25,    name='threshold1')
            t2, t2_op = tf.metrics.percentage_below(delta, 1.25**2, name='threshold2')
            t3, t3_op = tf.metrics.percentage_below(delta, 1.25**3, name='threshold3')
            tf.summary.scalar('threshold1', t1_op)
            tf.summary.scalar('threshold2', t2_op)
            tf.summary.scalar('threshold3', t3_op)
