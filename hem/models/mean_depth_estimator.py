from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import math
import hem



class mean_depth_estimator(hem.ModelPlugin):
    name = 'mean_depth_estimator'

    @staticmethod
    def arguments():
        args = {
            '--m_arch': {
                'type': str,
                'default': 'E2',
                'help': 'Architecture to use. Options are...'
                },
            '--examples': {
                'type': int,
                'default': 64,
                'help': 'Number of image summary examples to use.'}
            }
        return args


    @hem.default_to_cpu
    def __init__(self, x_y, args):
        # init/setup
        m_opt = hem.init_optimizer(args)
        m_tower_grads = []
        global_step = tf.train.get_global_step()

        # foreach gpu...
        for x_y, scope, gpu_id in hem.tower_scope_range(x_y, args.n_gpus, args.batch_size):

            # for i in range(len(x_y)):
            #     print('estimator', i, x_y[i])

            # x = x_y[0]
            # y = x_y[1]

            m_arch = {'E2': mean_depth_estimator.E2}
            x = x_y[4]
            x = tf.reshape(x, (-1, 3, 53, 70))
            # print('estimator x shape', x)
            y = x_y[5]

            with tf.variable_scope('model'):
                m_func = m_arch[args.m_arch]
                m = m_func(x, args, reuse=(gpu_id>0))
                self.output_layer = m
            # calculate losses
            m_loss = mean_depth_estimator.loss(m, x, y, args, reuse=(gpu_id > 0))
            # calculate gradients
            m_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')
            m_tower_grads.append(m_opt.compute_gradients(m_loss, var_list=m_params))
            # only need one batchnorm update (ends up being updates for last tower)
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
            # TODO: do we need to do this update for batchrenorm? for instance renorm?

        # average and apply gradients
        m_grads = hem.average_gradients(m_tower_grads, check_numerics=args.check_numerics)
        m_apply_grads = m_opt.apply_gradients(m_grads, global_step=global_step)

        # add summaries
        hem.summarize_losses()
        hem.summarize_gradients(m_grads, name='m_gradients')
        hem.summarize_layers('m_activations', [l for l in tf.get_collection('conv_layers') if 'model' in l.name], montage=True)
        mean_depth_estimator.montage_summaries(x, y, m, args)
        # improved_sampler.montage_summarpies(x, y, g, x_sample, y_sample, g_sampler, x_noise, g_noise, x_shuffled, y_shuffled, g_shuffle, args)
        # improved_sampler.sampler_summaries(y_sample, g_sampler, g_noise, y_shuffled, g_shuffle, args)

        # training ops
        with tf.control_dependencies(batchnorm_updates):
            self.m_train_op = m_apply_grads
        self.all_losses = hem.collection_to_dict(tf.get_collection('losses'))

    def train(self, sess, args, feed_dict):
        # _ = sess.run(self.d_train_op, feed_dict=feed_dict)
        # _ = sess.run(self.g_train_op, feed_dict=feed_dict)
        # results = sess.run(self.all_losses, feed_dict=feed_dict)
        _, results = sess.run([self.m_train_op, self.all_losses], feed_dict=feed_dict)
        return results

        # _, _, results = sess.run([self.d_train_op, self.g_train_op, self.all_losses], feed_dict=feed_dict)
        # return results




    @staticmethod
    def E2(x, args, reuse=False):
        with arg_scope([hem.conv2d],
                       reuse=reuse,
                       stride=2,
                       padding='SAME',
                       filter_size=5,
                       init=tf.contrib.layers.xavier_initializer,
                       activation=tf.nn.relu):          # 427 x 561 x 3
            l1 = hem.conv2d(x,     3,   64, name='l1')  # 214 x 281 x 64
            # print('l1', l1)
            l2 = hem.conv2d(l1,   64,  128, name='l2')  # 107, 141 x 128
            # print('l2', l2)
            l3 = hem.conv2d(l2,  128,  256, name='l3')  # 54 x 71 x 256
            # print('l3', l3)
            l4 = hem.conv2d(l3,  256,  512, name='l4')  # 27 x 36 x 512
            # print('l4', l4)
            l5 = hem.conv2d(l4,  512, 1024, name='l5')  # 14 x 18 x 1024
            # print('l5', l5)
            l6 = hem.conv2d(l5, 1024, 2048, name='l6')  # 4 x 5 x 2048
            # print('l6', l6)
            f = hem.flatten(l6) # 4096
            # print('f', f)
            l7 = hem.dense(f, 4096, 2048, name='l7') # 2048
            # print('l7', l7)
            l8 = hem.dense(l7, 2048, 1, activation=tf.nn.sigmoid, name='l8')  # 1
            # print('l8', l8)
        return l8


        #     l7 = hem.conv2d(l6, 2048, 2048, name='l7')  # 2 x 3 x 2048
        #     print('l7', l7)
        #     l8 = hem.conv2d(l7, 2048, 2048, name='l8')  # 1 x 2 x 2048
        #     print('l8', l8)
        #     f = hem.flatten(l8) # 4096
        #     print('f', f)
        #     l9 = hem.dense(f, 4096, 2048, name='l9') # 2048
        #     print('l9', l9)
        #     l10 = hem.dense(l9, 2048, 1, activation=tf.nn.sigmoid, name='l10')  # 1
        #     print('l10', l10)
        # return l10

    @staticmethod
    def loss(m, x, y, args, reuse=False):
        mean_depth = tf.reduce_mean(y, axis=[2, 3])
        l = tf.reduce_mean(tf.sqrt(tf.square(mean_depth - m)))
        # mean_depth = tf.one_hot(tf.reduce_mean(y), depth=2048)
        # depth_predictions = tf.one_hot(tf.argmax(m, dimension=1), depth=2048)
        # print('mean_depth:', mean_depth)
        # print('depth_predictions:', depth_predictions)
        # l = mean_depth - depth_predictions
        if not reuse:
            hem.add_to_collection('losses', [l])
        return l

    # # summaries
    # ############################################

    @staticmethod
    def montage_summaries(x, y, m, args):
        n = math.floor(math.sqrt(args.examples))
        with tf.variable_scope('model'):
            with arg_scope([hem.montage],
                           num_examples=args.examples,
                           height=n,
                           width=n,
                           colorize=True):
                hem.montage(y, name='real_depths')
                hem.montage(x, name='real_images')
                hem.montage(tf.expand_dims(tf.expand_dims(tf.reduce_mean(y, axis=[2, 3]), -1), -1), name='real_average_depths')
                hem.montage(tf.expand_dims(tf.expand_dims(m, -1), -1), name='predicted_average_depths')

    #
    # @staticmethod
    # def montage_summaries(x, y, g,
    #                       x_sample, y_sample, g_sampler,
    #                       x_noise, g_noise,
    #                       x_shuffled, y_shuffled, g_shuffle,
    #                       args):
    #     n = math.floor(math.sqrt(args.examples))
    #
    #     if args.g_arch in ['C1', 'D1']:
    #         x, x_loc, y_loc = tf.split(x, num_or_size_splits=[3, 1, 1], axis=1)
    #         x_noise, x_loc_noise, y_loc_noise = tf.split(x_noise, num_or_size_splits=[3, 1, 1], axis=1)
    #         x_shuffled, x_loc_shuffled, y_loc_shuffled = tf.split(x_shuffled, num_or_size_splits=[3, 1, 1], axis=1)
    #         x_sample, x_loc_sample, y_loc_sample = tf.split(x_sample, num_or_size_splits=[3, 1, 1], axis=1)
    #         with tf.variable_scope('model'):
    #             hem.montage(x_loc, num_examples=args.examples, height=n, width=n, name='x_loc', colorize=True)
    #             hem.montage(y_loc, num_examples=args.examples, height=n, width=n, name='y_loc', colorize=True)
    #
    #     if args.g_arch in ['E1']:
    #         x, x_loc, y_loc, mean_vec = tf.split(x, num_or_size_splits=[3, 1, 1, 1], axis=1)
    #         x_noise, x_loc_noise, y_loc_noise, mean_vec_noise = tf.split(x_noise, num_or_size_splits=[3, 1, 1, 1], axis=1)
    #         x_shuffled, x_loc_shuffled, y_loc_shuffled, mean_vec_shuffled = tf.split(x_shuffled, num_or_size_splits=[3, 1, 1, 1], axis=1)
    #         x_sample, x_loc_sample, y_loc_sample, mean_vec_sample = tf.split(x_sample, num_or_size_splits=[3, 1, 1, 1], axis=1)
    #         with tf.variable_scope('model'):
    #             hem.montage(x_loc, num_examples=args.examples, height=n, width=n, name='x_loc', colorize=True)
    #             hem.montage(y_loc, num_examples=args.examples, height=n, width=n, name='y_loc', colorize=True)
    #             hem.montage(mean_vec, num_examples=args.examples, height=n, width=n, name='mean_vec', colorize=True)
    #
    #     with tf.variable_scope('montage_preprocess'):
    #         g = tf.reshape(g, tf.shape(y))  # reattach
    #         g = hem.rescale(g, (-1, 1), (0, 1))
    #         y = hem.rescale(y, (-1, 1), (0, 1))
    #         g_sampler = tf.reshape(g_sampler, tf.shape(y))  # reattach
    #         g_sampler = hem.rescale(g_sampler, (-1, 1), (0, 1))
    #         y_sample = hem.rescale(y_sample, (-1, 1), (0, 1))
    #
    #     # add image montages
    #     with arg_scope([hem.montage],
    #                    num_examples=args.examples,
    #                    height=n,
    #                    width=n,
    #                    colorize=True):
    #         with tf.variable_scope('model'):
    #             hem.montage(x, name='images')
    #             hem.montage(y, name='real_depths')
    #             hem.montage(g, name='fake_depths')
    #         with tf.variable_scope('sampler'):
    #             hem.montage(x_sample, name='images')
    #             hem.montage(y_sample, name='real_depths')
    #             hem.montage(g_sampler, name='fake_depths')
    #         with tf.variable_scope('shuffled'):
    #             hem.montage(x_shuffled, name='images')
    #             hem.montage(g_shuffle, name='fake_depths')
    #         with tf.variable_scope('noise'):
    #             hem.montage(x_noise, name='images')
    #             hem.montage(g_noise, name='fake_depths')
    #
