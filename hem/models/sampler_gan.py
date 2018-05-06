"""Implementation of sampler GAN.

Supports multi-GPU training.

Notation conventions:

x            = Tensor, real image (RGB)
y            = Tensor, real depth map (single channel)
z            = noise vector
G            = Generator
D            = Discriminator
y_hat        = G(x, z)
h            = D(x, y)
h_hat        = D(x, y_hat)
x_sample     = single repeated image from x
y_sample     = single repeated depth map from y
y_hat_sample = estimated depth maps for x

However, x is also used anywhere the input is generic.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import math
import hem


def center_crop(x, fraction):
    x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.map_fn(lambda a: tf.image.central_crop(a, central_fraction=fraction), x)
    return tf.transpose(x, [0, 3, 1, 2])


class sampler_gan(hem.ModelPlugin):
    name = 'sampler_gan'

    @staticmethod
    def arguments():
        args = {
            '--batch_norm_disc': {
                'action': 'store_true',
                'default': False,
                'help': 'Whether to use batch_norm in the discriminator.'},
            '--batch_norm_gen': {
                'action': 'store_true',
                'default': False,
                'help': 'Whether to use batch norm in the generator.'},
            '--examples': {
                'type': int,
                'default': 64,
                'help': 'Number of image summary examples to use.'},
            '--garch': {
                'default': 'large',
                'choices': ['small','large'],
                'help': 'Alternative generator architecture options.'},
            '--darch': {
                'default': 'early',
                'choices': ['early', 'late'],
                'help': 'Discriminator architecture: whether it should concat early or late.'
                }
            }
        return args

    @hem.default_to_cpu
    def __init__(self, x_y, args):
        """Create sampler GAN model on the graph.

        Args:
            x: Tensor, the real images.
            args: Argparse structure

                Returns:
            Function, the training function. Call for one iteration of training.
        """
        # init/setup
        g_opt = hem.init_optimizer(args)
        d_opt = hem.init_optimizer(args)
        g_tower_grads = []
        d_tower_grads = []
        global_step = tf.train.get_global_step()

        # foreach gpu...
        for x_y, scope, gpu_id in hem.tower_scope_range(x_y, args.n_gpus, args.batch_size):
            # split inputs and rescale
            x = hem.rescale(x_y[0], (0, 1), (-1, 1))
            y = hem.rescale(x_y[1], (0, 1), (-1, 1))
            # center-crop depth to 31x31
            y = center_crop(y, 0.4769)
            y = tf.reshape(y, (-1, 1, 31, 31))
            # repeated image tensor for sampling
            x_sample = tf.stack([x[0]] * args.examples)
            y_sample = tf.stack([y[0]] * args.examples)

            # create model
            with tf.variable_scope('generator'):
                g = sampler_gan.generator(x, args, reuse=(gpu_id > 0))
                g_sampler = sampler_gan.generator(x_sample, args, reuse=True)
            with tf.variable_scope('discriminator'):
                # discriminator
                d_real, d_real_logits = sampler_gan.discriminator(x, y, args, reuse=(gpu_id > 0))
                d_fake, d_fake_logits = sampler_gan.discriminator(x, g, args, reuse=True)

            # losses
            g_loss, d_loss = sampler_gan.loss(d_real, d_real_logits,
                                              d_fake, d_fake_logits,
                                              g, y, args, reuse=(gpu_id > 0))
            # gradients
            g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
            d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
            g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_params))
            d_tower_grads.append(d_opt.compute_gradients(d_loss, var_list=d_params))
            # only need one batchnorm update (ends up being updates for last tower)
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        # average and apply gradients
        g_grads = hem.average_gradients(g_tower_grads, check_numerics=args.check_numerics)
        d_grads = hem.average_gradients(d_tower_grads, check_numerics=args.check_numerics)
        g_apply_grads = g_opt.apply_gradients(g_grads, global_step=global_step)
        d_apply_grads = d_opt.apply_gradients(d_grads, global_step=global_step)

        # add summaries
        sampler_gan.montage_summaries(x, y, g, args, x_sample, y_sample, g_sampler)
        sampler_gan.activation_summaries()
        sampler_gan.loss_summaries()
        sampler_gan.gradient_summaries(g_grads, 'generator_gradients')
        sampler_gan.gradient_summaries(d_grads, 'discriminator_gradients')
        sampler_gan.sampler_summaries(y_sample, g_sampler, args)

        # training ops
        with tf.control_dependencies(batchnorm_updates):
            self.d_train_op = d_apply_grads
            self.g_train_op = g_apply_grads
        self.all_losses = hem.collection_to_dict(tf.get_collection('losses'))


    def train(self, sess, args, feed_dict):
        sess.run(self.d_train_op, feed_dict=feed_dict)
        sess.run(self.g_train_op, feed_dict=feed_dict)
        results = sess.run(self.all_losses, feed_dict=feed_dict)
        return results


    @staticmethod
    def generator(x, args, reuse=False):
        """Adds generator nodes to the graph using input `x`.

        This is an encoder-decoder architecture.

        Args:
            x: Tensor, the input image batch (65x65 RGB images)
            args: Argparse struct.
            reuse: Bool, whether to reuse variables.

        Returns:
             Tensor, the generator's output, the estimated depth map
             (33x33 estimated depth map)
        """
        # encoder
        with tf.variable_scope('encoder', reuse=reuse),\
             arg_scope([hem.conv2d],
                       reuse=reuse,
                       use_batch_norm=args.batch_norm_gen,
                       filter_size=5,
                       stride=2,
                       padding='VALID',
                       init=tf.contrib.layers.xavier_initializer,
                       # init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                       activation=tf.nn.relu):
                       # activation=lambda x: hem.lrelu(x, leak=0.2)):                               # 65x65x3
            noise = tf.random_uniform([args.batch_size, 1, 65, 65], minval=-1.0, maxval=1.0)
            x = tf.concat([x, noise], axis=1)                                                      # 65x65x4
            e1 = hem.conv2d(x, 4, 64, name='1', use_batch_norm=False)                              # 31x31x64
            if args.garch == 'large':
                e1 = hem.conv2d(e1, 64, 64, stride=1, padding='SAME', name='1b')                   # 31x31x64
                e1 = hem.conv2d(e1, 64, 64, stride=1, padding='SAME', name='1c')                   # 31x31x64
            e2 = hem.conv2d(e1,  64, 128, name='2')                                                # 14x14x128
            if args.garch == 'large':
                e2 = hem.conv2d(e2,  128, 128, stride=1, padding='SAME', name='2b')                # 14x14x128
                e2 = hem.conv2d(e2,  128, 128, stride=1, padding='SAME', name='2c')                # 14x14x128
            e3 = hem.conv2d(e2, 128, 256, name='3')                                                # 5x5x256
            if args.garch == 'large':
                e3 = hem.conv2d(e3, 256, 256, stride=1, padding='SAME', name='3b')                 # 5x5x256
                e3 = hem.conv2d(e3, 256, 256, stride=1, padding='SAME', name='3c')                 # 5x5x256
            e4 = hem.conv2d(e3, 256, 512, name='4')                                                # 1x1x512
        # decoder
        with tf.variable_scope('decoder', reuse=reuse),\
             arg_scope([hem.deconv2d, hem.conv2d],
                       reuse=reuse,
                       use_batch_norm=args.batch_norm_gen,
                       filter_size=5,
                       stride=2,
                       init=tf.contrib.layers.xavier_initializer,
                       # init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                       padding='VALID',
                       activation=lambda x: hem.lrelu(x, leak=0.2)):                                 # 1x1x512
            # TODO figure out cleaner way to add skip layers
            y = hem.deconv2d(e4,  512, 256, output_shape=(args.batch_size, 256,  5,  5), name='1')      # 5x5x256
            y = tf.concat([y, e3], axis=1)                                                             # 5x5x512
            if args.garch == 'large':
                y = hem.deconv2d(y, 512, 512, output_shape=(args.batch_size, 512, 5, 5),
                                 stride=1, padding='SAME', name='1b')

            y = hem.deconv2d(y,  512, 128, output_shape=(args.batch_size, 128, 14, 14), name='2')      # 14x14x128
            y = tf.concat([y, e2], axis=1)                                                             # 14x14x256
            if args.garch == 'large':
                y = hem.deconv2d(y, 256, 256, output_shape=(args.batch_size, 256, 14, 14),
                                 stride=1, padding='SAME', name='2b')
            y = hem.deconv2d(y,  256,  64, output_shape=(args.batch_size,  64, 31, 31), name='3')      # 31x31x64
            y = tf.concat([y, e1], axis=1)  # 31x31x128
            if args.garch == 'large':
                y = hem.deconv2d(y, 128, 128, output_shape=(args.batch_size, 128, 31, 31),
                                 stride=1, padding='SAME', name='3b')  # 31x31x64
            y = hem.conv2d(y,    128,   1, stride=1, padding='SAME', activation=tf.nn.tanh, name='7')  # 31x31x1
        return y


    @staticmethod
    def discriminator(x, y, args, reuse=False):
        """Adds (PatchGAN) discriminator nodes to the graph, given RGB and D inputs.

        Args:
            x_rgb: Tensor, the RGB component of the real or fake images.
            x_depth: Tensor, the depth component of the real or fake images.
            args: Argparse struct.
            reuse: Bool, whether to reuse variables.

        Returns:
            Tensor, contains 70x70 patches with a probability assigned to each.
        """
        with arg_scope([hem.conv2d],
                       reuse=reuse,
                       use_batch_norm=args.batch_norm_disc,
                       activation=lambda x: hem.lrelu(x, leak=0.2),
                       init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                       padding='VALID',
                       filter_size=5,
                       stride=2):
            if args.darch == 'early':
                rgb_path = hem.conv2d(x, 3, 64, name='rgb_path', use_batch_norm=False)  # 65x65x3 -> 31x31x64
                depth_path = hem.conv2d(y, 1, 64, stride=1, padding='SAME', name='depth_path', use_batch_norm=False) # 31x31x64
                h = tf.concat([rgb_path, depth_path], axis=1)   # 31x31x128
                h = hem.conv2d(h, 128, 256, name='h1')          # 14x14x256
                h = hem.conv2d(h, 256, 512, name='h2')          # 5x5x512
                h = hem.conv2d(h, 512, 512, name='h3', activation=None)          # 1x1x512
            elif args.darch == 'late':
                h1 = hem.conv2d(x,    3, 64, name='h1.a', use_batch_norm=False)
                h1 = hem.conv2d(h1,  64, 128, name='h1.b')
                h1 = hem.conv2d(h1, 128, 256, name='h1.c')
                h1 = hem.conv2d(h1, 256, 512, name='h1.d')  # 1x1x512 ?
                h2 = hem.conv2d(y,    1, 64, name='h2.a', stride=1, padding='SAME', use_batch_norm=False)
                h2 = hem.conv2d(h2,  64, 128, name='h2.b')
                h2 = hem.conv2d(h2, 128, 256, name='h2.c')
                h2 = hem.conv2d(h2, 256, 512, name='h2.d')  # 1x1x512 ?
                h = tf.concat([h1, h2], axis=1) # 1x1x1024
                h = hem.conv2d(h, 1024, 1024, stride=1, padding='SAME', name='h.a')
                h = hem.conv2d(h, 1024, 512, name='h.b', filter_size=1)
        # layer outout, logits
        return tf.nn.sigmoid(h), h


    @staticmethod
    def loss(d_real, d_real_logits, d_fake, d_fake_logits, g, x_depth, args, reuse=False):
        """Adds loss nodes to the graph.

        Args:
            d_real: Tensor, the discriminator's output with a real batch.
            d_fake: Tensor, the discriminator's output with a fake batch.
            g: Tensor, the generator's output.
            x_depth: Tensor, the real depth maps.
            reuse: Bool, whether to add the loss nodes to the loss collection
                   for later summary collection. Should only be True once.

        Returns:
            Tensors, the losses for the generator and discriminator, respectively.
        """
        def xentropy(logits, labels):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        with tf.variable_scope('loss'):
            g = hem.rescale(g, (-1, 1), (0, 1))
            x_depth = hem.rescale(x_depth, (-1, 1), (0, 1))
            # losses
            with tf.variable_scope('generator'):
                g_fake = tf.reduce_mean(xentropy(d_fake_logits, tf.ones_like(d_fake)), name='g_fake')
            with tf.variable_scope('discriminator'):
                d_real = tf.reduce_mean(xentropy(d_real_logits, tf.ones_like(d_fake)), name='d_real')
                d_fake = tf.reduce_mean(xentropy(d_fake_logits, tf.zeros_like(d_fake)), name='d_fake')
                d_total = tf.identity(d_real + d_fake, name='total')
            rmse_loss = hem.rmse(x_depth, g)
            l1_loss = tf.reduce_mean(tf.abs(x_depth - g), name='l1')

        # only add these to the collection once
        if not reuse:
            hem.add_to_collection('losses', [g_fake, d_real, d_fake, d_total, rmse_loss, l1_loss])
        return g_fake, d_total


    # summaries
    ############################################

    @staticmethod
    def montage_summaries(x_rgb, x_depth, g, args, x_repeated_rgb, x_repeated_depth, g_sampler):
        """Adds montage summaries to the graph for the images, generator, and sampler.

        Args:
            x_rgb: Tensor, the real RGB image input batch.
            x_depth: Tensor, the real depth input batch.
            g: Tensor, the generator's output.
            args: Argparse struct.
            x_repeated_rgb: Tensor, the real RGB image inputs to the sampler path.
            x_repeated_depth: Tensor, the real RGB depth inputs to the sampler path.
            g_sampler: Tensor, the sampler generator outputs.

        Returns:
            None.
        """
        # print('x shape', x_rgb.shape)
        # print('y shape', x_depth.shape)
        # print('y_hat shape', g.shape)
        # print('x_repeated shape', x_repeated_depth.shape)
        # print('y_hat_repeated shape', g_sampler.shape)

        # example image montages
        n = math.floor(math.sqrt(args.examples))
        # rescale
        with tf.variable_scope('montage_preprocess'):
            g = tf.reshape(g, tf.shape(x_depth))  # re-attach lost shape info
            g = hem.rescale(g, (-1, 1), (0, 1))
            x_depth = hem.rescale(x_depth, (-1, 1), (0, 1))
            g_sampler = tf.reshape(g_sampler, tf.shape(x_depth))
            g_sampler = hem.rescale(g_sampler, (-1, 1), (0, 1))
            x_repeated_depth = hem.rescale(x_repeated_depth, (-1, 1), (0, 1))

            # x_depth = tf.reshape(x_depth, (-1, 1, 65, 65))

        # add image montages
        with arg_scope([hem.montage],
                       height=n,
                       width=n):
            # example batches
            hem.montage(x_rgb[0:args.examples],   name='real/images')
            hem.montage(x_depth[0:args.examples], name='real/depths', colorize=True)
            hem.montage(x_rgb[0:args.examples],   name='fake/images')
            hem.montage(g[0:args.examples],       name='fake/depths', colorize=True)
            # sampler
            hem.montage(x_repeated_rgb[0:args.examples], name='sampler/images')
            hem.montage(x_repeated_depth[0:args.examples], name='sampler/depths', colorize=True)
            hem.montage(g_sampler[0:args.examples], name='sampler/fake', colorize=True)

    @staticmethod
    def summarize_moments(x, name, args):
        mean, var = tf.nn.moments(x, axes=[0])
        tf.summary.scalar(name + '/mean', tf.reduce_mean(mean))
        tf.summary.scalar(name + '/var', tf.reduce_mean(var))
        tf.summary.histogram(name + '/mean', mean)
        tf.summary.histogram(name + '/var', var)
        var = tf.expand_dims(var, axis=0)
        var = hem.colorize(var)
        tf.summary.image(name + '/var', var)

    @staticmethod
    def sampler_summaries(x_repeated_depth, g_sampler, args):
        with tf.variable_scope('sampler_metrics'):
            # mean and var metrics for sampler
            g_sampler = tf.reshape(g_sampler, x_repeated_depth.shape) # reattach shape info
            sampler_gan.summarize_moments(g_sampler, 'depth', args)
            sampler_gan.summarize_moments(g_sampler - tf.reduce_mean(g_sampler), 'depth_normalized', args)
            # ground truth example
            ic = tf.expand_dims(x_repeated_depth[0], axis=0)
            ic = hem.rescale(ic, (-1, 1), (0, 1))
            ic = hem.colorize(ic)
            tf.summary.image('real_depth', ic)
            # mean and min l2 loss from sampler
            sample_l2_loss = tf.reduce_mean(tf.square(x_repeated_depth - g_sampler), axis=[1, 2, 3])
            mean_l2_loss = tf.reduce_mean(sample_l2_loss)
            min_l2_loss = tf.reduce_min(sample_l2_loss)
            tf.summary.scalar('mean sample l2', mean_l2_loss)
            tf.summary.scalar('min sample l2', min_l2_loss)
            sample_rmse_loss = tf.reduce_mean(tf.sqrt(tf.square(x_repeated_depth - g_sampler)), axis=[1, 2, 3])
            mean_rmse_loss = tf.reduce_mean(sample_rmse_loss)
            min_rmse_loss = tf.reduce_min(sample_rmse_loss)
            tf.summary.scalar('mean sample rmse', mean_rmse_loss)
            tf.summary.scalar('min sample rmse', min_rmse_loss)

    @staticmethod
    def gradient_summaries(grads, name='gradients'):
        with tf.variable_scope(name):
            for g, v in grads:
                n = hem.tensor_name(v)
                tf.summary.histogram(n, g)
                tf.summary.scalar(n, tf.reduce_mean(g))

    @staticmethod
    def loss_summaries():
        for l in tf.get_collection('losses'):
            tf.summary.scalar(hem.tensor_name(l), l)
            tf.summary.histogram(hem.tensor_name(l), l)

    @staticmethod
    def activation_summaries():
        layers = tf.get_collection('conv_layers')
        d_layers = [l for l in layers if 'discriminator' in l.name]
        g_layers = [l for l in layers if 'generator' in l.name]
        with tf.variable_scope('discriminator_activations'):
            for l in d_layers:
                layer_name = hem.tensor_name(l)
                layer_name = '/'.join(layer_name.split('/')[0:2])
                l = tf.transpose(l, [0, 2, 3, 1])
                tf.summary.histogram(layer_name, l)
                tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(l))
                tf.summary.scalar(layer_name + '/mean', tf.reduce_mean(l))
        with tf.variable_scope('generator_activations'):
            for l in g_layers:
                layer_name = hem.tensor_name(l)
                layer_name = '/'.join(layer_name.split('/')[0:2])
                l = tf.transpose(l, [0, 2, 3, 1])
                tf.summary.histogram(layer_name, l)
                tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(l))
                tf.summary.scalar(layer_name + '/mean', tf.reduce_mean(l))

    @staticmethod
    def generate_visualizations(event_dir, output_dir):
        pass













