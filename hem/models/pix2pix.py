"""Implementation of canonical 'pix2pix' Conditional GAN.

Supports multi-GPU training.

Notation conventions:

x            = Tensor, RGB channels of real images
y            = Tensor, depth of real images
x_y          = Tensor, the combined RGB-D real images
x_sample     = Tensor, RGB channels of a single real image, repeated n times
y_sample     = Tensor, depth of a single real image, repeated n times

y_hat        = Tensor, generator's output (i.e., depth estimates)
y_hat_sample = Tensor, generator's output from input x_prime (i.e., depth estimates of sampler)
h_real       = Tensor, the discriminator's output given real inputs
h_fake       = Tensor, the discriminator's output given fake inputs

However, x is also used anywhere the input is generic.

Sources:
-------
- `Image-to-Image Translation with Conditional Adversarial Networks`
  http://arxiv.org/abs/1611.07004
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import math
import hem


class pix2pix(hem.ModelPlugin):
    name = 'pix2pix'

    @staticmethod
    def arguments():
        args = {
            '--skip_layers': {
                'action': 'store_true',
                'default': 'false',
                'help': 'Adds skip layers to the generator.'},
            '--noise': {
                'type': str,
                'nargs': '*',
                'choices': ['input','latent','end'],
                'default': [],
                'help': 'Inject noise into the generator at the specified point.'},
            '--dropout': {
                'type': float,
                'default': 0,
                'help':'Whether to use dropout on the early decoder layers.'},
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
            '--n_disc_train': {
                'type': int,
                'default': 1,
                'help': 'Number of times to train the discriminator before training the generator.'},
            '--add_l1': {
                'action': 'store_true',
                'default': False,
                'help': "Adds L1 reconstruction loss to the generator's loss function."},
            '--lambda': {
                'type': float,
                'default': 10.0,
                'help': "Regularization term for generator loss (used in conjunction with l1)."}
            }
        return args

    @hem.default_to_cpu
    def __init__(self, x_y, args):
        """Create conditional GAN ('pix2pix') model on the graph.

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
        # rescale to [-1, 1]
        # x_y = hem.rescale(x_y, (0, 1), (-1, 1))

        # foreach gpu...
        for x_y, scope, gpu_id in hem.tower_scope_range(x_y, args.n_gpus, args.batch_size):
            # split inputs and scale to [-1, 1]
            x = hem.rescale(x_y[0], (0, 1), (-1, 1))
            y = hem.rescale(x_y[1], (0, 1), (-1, 1))

            # x, y = tf.split(x_y, num_or_size_splits=[3, 1], axis=1)
            # repeated image tensor for sampling
            x_sample = tf.stack([x[0]] * args.examples)
            y_sample = tf.stack([y[0]] * args.examples)

            # create model
            with tf.variable_scope('generator'):
                g = pix2pix.generator(x, args, reuse=(gpu_id > 0))
                g_sampler = pix2pix.generator(x_sample, args, reuse=True)
            with tf.variable_scope('discriminator'):
                d_real, d_real_logits = pix2pix.discriminator(x, y, args, reuse=(gpu_id > 0))
                d_fake, d_fake_logits = pix2pix.discriminator(x, g, args, reuse=True)

            # losses
            g_loss, d_loss = pix2pix.loss(d_real, d_real_logits, d_fake, d_fake_logits, g, y, args, reuse=(gpu_id > 0))
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
        pix2pix.montage_summaries(x, y, g, args, x_sample, y_sample, g_sampler, d_real, d_fake)
        pix2pix.activation_summaries()
        pix2pix.loss_summaries()
        pix2pix.gradient_summaries(g_grads, 'generator_gradients')
        pix2pix.gradient_summaries(d_grads, 'discriminator_gradients')
        pix2pix.sampler_summaries(y_sample, g_sampler, args)


        # training ops
        with tf.control_dependencies(batchnorm_updates):
            self.d_train_op = d_apply_grads
            self.g_train_op = g_apply_grads
        self.all_losses = hem.collection_to_dict(tf.get_collection('losses'))


    def train(self, sess, args, feed_dict):
        for x in range(args.n_disc_train):
            sess.run(self.d_train_op, feed_dict=feed_dict)
        sess.run(self.g_train_op, feed_dict=feed_dict)
        results = sess.run(self.all_losses, feed_dict=feed_dict)
        return results


    @staticmethod
    def generator(x, args, reuse=False):
        """Adds generator nodes to the graph using input `x`.

        This is an encoder-decoder architecture.

        Args:
            x: Tensor, the input image batch (256x256 RGB images)
            args: Argparse struct.
            reuse: Bool, whether to reuse variables.

        Returns:
             Tensor, the generator's output, the estimated depth map
             (256x256 grayscale).
        """
        # encoder
        with arg_scope([hem.conv2d],
                       reuse=reuse,
                       use_batch_norm=args.batch_norm_gen,
                       filter_size=4,
                       stride=2,
                       init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                       activation=lambda x: hem.lrelu(x, leak=0.2)):
            with tf.variable_scope('enocder', reuse=reuse):
                if 'input' in args.noise:
                    noise = tf.random_uniform([args.batch_size, 1, 256, 256], minval=-1.0, maxval=1.0)
                    e1 = hem.conv2d(tf.concat([x, noise], axis=1), 4, 64, name='1', use_batch_norm=False)  # 128x128x64
                else:
                    e1 = hem.conv2d(x, 3, 64, name='1', use_batch_norm=False)
                e2 = hem.conv2d(e1, 64, 128, name='2')  # 64x64x128
                e3 = hem.conv2d(e2, 128, 256, name='3')  # 32x32x256
                e4 = hem.conv2d(e3, 256, 512, name='4')  # 16x16x512
                e5 = hem.conv2d(e4, 512, 512, name='5')  # 8x8x512
                e6 = hem.conv2d(e5, 512, 512, name='6')  # 4x4x512
                e7 = hem.conv2d(e6, 512, 512, name='7')  # 2x2x512
                e8 = hem.conv2d(e7, 512, 512, name='8')  # 1x1x512
        # decoder
        with arg_scope([hem.deconv2d, hem.conv2d],
                       reuse=reuse,
                       use_batch_norm=True,
                       filter_size=4,
                       stride=2,
                       init=lambda: tf.random_normal_initializer(mean=0, stddev=0.02),
                       activation=lambda x: hem.lrelu(x, leak=0)):
            # TODO figure out cleaner way to add skip layers
            with tf.variable_scope('decoder', reuse=reuse):
                if 'latent' in args.noise:
                    noise = tf.random_uniform([args.batch_size, 512, 1, 1], minval=-1.0, maxval=1.0)
                    y = hem.deconv2d(tf.concat([e8, noise], axis=1), 1024, 512, name='1', dropout=args.dropout)  # 2x2x512*2
                else:
                    y = hem.deconv2d(e8, 512, 512, name='1', dropout=args.dropout)  # 2x2x512*2
                y = tf.concat([y, e7], axis=1)
                y = hem.deconv2d(y, 1024, 512, name='2', dropout=args.dropout)  # 4x4x512*2
                y = tf.concat([y, e6], axis=1)
                y = hem.deconv2d(y, 1024, 512, name='3', dropout=args.dropout)  # 8x8x512*2
                y = tf.concat([y, e5], axis=1)
                y = hem.deconv2d(y, 1024, 512, name='4')  # 16x16x512*2
                y = tf.concat([y, e4], axis=1)
                y = hem.deconv2d(y, 1024, 256, name='5')  # 32x32x256*2
                y = tf.concat([y, e3], axis=1)
                y = hem.deconv2d(y, 512, 128, name='6')  # 64x64x128*2
                y = tf.concat([y, e2], axis=1)
                y = hem.deconv2d(y, 256, 64, name='7')  # 128x128x64*2
                y = tf.concat([y, e1], axis=1)
                if 'end' in args.noise:
                    noise = tf.random_uniform([args.batch_size, 1, 128, 128], minval=-1.0, maxval=1.0)
                    y = hem.deconv2d(tf.concat([y, noise], axis=1), 129, 1, name='8', activation=tf.nn.tanh)  # 256x256x1
                else:
                    y = hem.deconv2d(y, 128, 1, name='8', activation=tf.nn.tanh)  # 256x256x1
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
                       filter_size=4,
                       stride=2):
            x_y = tf.concat([x, y], axis=1)
            h = hem.conv2d(x_y, 4, 64, name='m1', use_batch_norm=False)
            h = hem.conv2d(h, 64, 128, name='m2')
            h = hem.conv2d(h, 128, 256, name='m3')
            h = hem.conv2d(h, 256, 512, name='m4')
            h = hem.conv2d(h, 512, 1, name='m5', activation=None)  # , activation=tf.nn.sigmoid)
            # hem.montage(tf.nn.sigmoid(h), )
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
        with tf.variable_scope('loss'):
            # rescale to [0, 1]
            g = hem.rescale(g, (-1, 1), (0, 1))
            x_depth = hem.rescale(x_depth, (-1, 1), (0, 1))
            # losses
            with tf.variable_scope('generator'):
                g_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake)),
                    name='g_fake')
                l_term = 10.0
                l1 = tf.reduce_mean(tf.abs(x_depth - g), name='l1')
                if args.add_l1:
                    g_fake = g_fake + l_term * l1
                g_total = tf.identity(g_fake, name='total')
            with tf.variable_scope('discriminator'):
                d_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real)),
                    name='d_real')
                d_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake)),
                    name='d_fake')
                d_total = tf.identity(d_real + d_fake, name='total')
            rmse_loss = hem.rmse(x_depth, g)

        # only add these to the collection once
        if not reuse:
            hem.add_to_collection('losses', [l1, g_fake, g_total, d_real, d_fake, d_total, rmse_loss])
        return g_total, d_total


    # summaries
    ############################################

    @staticmethod
    def montage_summaries(x_rgb, x_depth, g, args, x_repeated_rgb, x_repeated_depth, g_sampler, d_real, d_fake):
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
            # discriminator patches
            hem.montage(d_fake[0:args.examples], name='discriminator_patches/fake')
            hem.montage(d_real[0:args.examples], name='discriminator_patches/real')

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
            pix2pix.summarize_moments(g_sampler, 'depth', args)
            pix2pix.summarize_moments(g_sampler - tf.reduce_mean(g_sampler), 'depth_normalized', args)
            # ground truth example
            ic = tf.expand_dims(x_repeated_depth[0], axis=0)
            ic = hem.rescale(ic, (-1, 1), (0, 1))
            ic = hem.colorize(ic)
            tf.summary.image('real_depth', ic)
            # mean and min l2 loss from sampler
            sample_l2_loss = tf.reduce_mean(tf.square(x_repeated_depth - g_sampler), axis=[1, 2, 3])
            mean_l2_loss = tf.reduce_mean(sample_l2_loss)
            min_l2_loss = tf.reduce_min(sample_l2_loss)
            tf.summary.scalar('mean sample l2_loss', mean_l2_loss)
            tf.summary.scalar('min sample l2_loss', min_l2_loss)

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
