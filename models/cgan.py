"""Implementation of 'pix2pix' Conditional GAN. Uses Improved Wasserstein methods.

Supports multi-GPU training.

Sources:
-------
- `Image-to-Image Translation with Conditional Adversarial Networks`
  http://arxiv.org/abs/1611.07004
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.arg_scope import arg_scope
import hem

# TODO add support for sizing other than 64x64
# TODO add support to hem ops for multiple input tensors


@hem.default_to_cpu
def cgan(x, args):
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
    x = hem.rescale(x, (0, 1), (-1, 1))

    for x, scope, gpu_id in hem.tower_scope_range(x, args.n_gpus, args.batch_size):
        # model
        x_rgb, x_depth = _split(x)
        # generator
        with tf.variable_scope('generator'):

            g = generator(x_rgb, args, reuse=(gpu_id > 0))
        # discriminator
        with tf.variable_scope('discriminator'):
            d_real = discriminator(x, args, reuse=(gpu_id > 0))
            d_fake = discriminator(g, args, reuse=True)
        # losses
        g_loss, d_loss = losses(x, g, d_fake, d_real, args, reuse=(gpu_id > 0))
        # gradients
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_tower_grads.append(g_opt.compute_gradients(g_loss, var_list=g_params))
        d_tower_grads.append(d_opt.compute_gradients(d_loss, var_list=d_params))
        # only need one batchnorm update (ends up being updates for last tower)
        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

    # average and apply gradients
    g_grads = hem.average_gradients(g_tower_grads)
    d_grads = hem.average_gradients(d_tower_grads)
    g_apply_grads = g_opt.apply_gradients(g_grads, global_step=global_step)
    d_apply_grads = d_opt.apply_gradients(d_grads, global_step=global_step)
    # add summaries
    _summaries(g, x, args)
    hem.add_basic_summaries(g_grads + d_grads)
    # training
    return _train_cgan(g_apply_grads, d_apply_grads, batchnorm_updates)


def _summaries(g, x, args):
    """Add specialized summaries to the graph.

    This adds summaries that:
        - Track variability in generator samples given random noise vectors.
        - Track how much of the noise vector is used.
        - Generate examples from the real and learned distributions.

    Args:
        g: Tensor, the generator's output. i.e., the fake images.
        x: Tensor, the real (input) images.
        args: Argparse structure.

    Returns:
        None
    """
    # 1. generate multiple samples using a single image
    with tf.variable_scope(tf.get_variable_scope()):
        gpu_id = 0
        with tf.device(hem.variables_on_cpu(gpu_id)):
            with tf.name_scope('tower_{}'.format(gpu_id)) as scope:
                # print('input shape', x.shape)
                with tf.variable_scope('generator'):
                    # using first image in batch, form new one with just this image
                    x_repeated = tf.stack([x[0]] * args.batch_size)
                    x_rgb, x_depth = _split(x_repeated)
                    # then create a new path for the generator using just this dataset
                    import copy
                    args_copy = copy.copy(args)
                    args_copy.batch_norm=False
                    d = generator(x_rgb, args_copy, reuse=True)
                    # scale to [0, 1]
                    sampler_rgb, sampler_depth = _split(d)
                    sampler_rgb = hem.rescale(sampler_rgb, (-1, 1), (0, 1))
                    sampler_depth = hem.rescale(sampler_depth, (-1, 1), (0, 1))

    with tf.variable_scope('sampler'):
        hem.montage(sampler_rgb[0:args.examples], 8, 8, name='images')
        hem.montage(sampler_depth[0:args.examples], 8, 8, name='depths')

        # and track the variance of the depth predictions
        mean, var = tf.nn.moments(sampler_depth, axes=[0])
        hem.scalars(('predicted_depth_mean', tf.reduce_mean(mean)), ('predicted_depth_var', tf.reduce_mean(var)))

        # and the images (temporary, for calibration/sanity check)
        x_rgb, x_depth = _split(x)
        mean, var = tf.nn.moments(x_depth, axes=[0])
        hem.scalars(('real_depth_mean', tf.reduce_mean(mean)), ('real_depth_var', tf.reduce_mean(var)))

        sampler_rgb = tf.transpose(sampler_rgb, [0, 2, 3, 1])
        sampler_depth = tf.transpose(sampler_depth, [0, 2, 3, 1])

        # mean, var = tf.nn.moments(tf.image.rgb_to_grayscale(sampler_rgb), axes=[0])
        axis = [0, -1]
        mean, var = tf.nn.moments(tf.image.rgb_to_grayscale(sampler_rgb), axes=[0])
        hem.scalars(('image_mean', tf.reduce_mean(mean)), ('image_var', tf.reduce_mean(var)))

        mean, var = tf.nn.moments(sampler_depth, axes=[0])
        hem.scalars(('depth_mean', tf.reduce_mean(mean)), ('depth_var', tf.reduce_mean(var)))


    # 2. generate summaries for real and fake images
    with tf.variable_scope('examples'):
        hem.histograms(('fake', g), ('real', x))
        # rescale, split, and colorize
        x = hem.rescale(x[0:args.examples], (-1, 1), (0, 1))
        g = hem.rescale(g[0:args.examples], (-1, 1), (0, 1))
        hem.histograms(('fake_rescaled', g), ('real_rescaled', x))

        x_rgb, x_depth = _split(x)
        g_rgb, g_depth = _split(g)
        # note: these are rescaled to (0, 1)
        hem.histograms(('real_depth', x_depth), ('fake_depth', g_depth),
                       ('real_rgb', x_rgb), ('fake_rgb', g_rgb))
        # add montages
        # TODO shouldn't be fixed to 8, but to ceil(sqrt(args.examples))
        hem.montage(x_rgb,   8, 8, name='real/images')
        hem.montage(x_depth, 8, 8, name='real/depths')
        hem.montage(g_rgb,   8, 8, name='fake/images')
        hem.montage(g_depth, 8, 8, name='fake/depths')

    # 3. add additional summaries for weights and biases in e_c1 (the initial noise layer)
    # TODO don't iterate through list, but grab directly by full name
    with tf.variable_scope('noise'):
        for l in tf.get_collection('weights'):
            if 'e_c1' in hem.tensor_name(l):
                print(l, hem.tensor_name(l))
                x_rgb, x_noise = _split(l)
                hem.histograms(('rgb/weights', x_rgb), ('noise/weights', x_noise))
                hem.scalars(('rgb/weights/sparsity', tf.nn.zero_fraction(x_rgb)),
                            ('noise/weights/sparsity', tf.nn.zero_fraction(x_noise)),
                            ('noise/weights/mean', tf.reduce_mean(x_noise)),
                            ('rgb/weights/mean', tf.reduce_mean(x_rgb)))
                break


def _train_cgan(g_apply_grads, d_apply_grads, batchnorm_updates):
    """Generates helper to train a Conditional, Improved Wasserstein GAN.

    Batchnorm updates are applied only to discriminator, since generator
    doesn't use batchnorm. Discriminator is trained to convergence
    before training the generator.

    Args:
        g_apply_grads:
        d_apply_grads:
        batchnorm_updates:

    Returns:
        Function, a function that trains the model for one iteration per call.
    """
    g_train_op = g_apply_grads
    with tf.control_dependencies(batchnorm_updates):
        d_train_op = d_apply_grads
    all_losses = hem.collection_to_dict(tf.get_collection('losses'))

    def helper(sess, args, handle, training_handle):
        for i in range(args.n_disc_train):
            sess.run(d_train_op, feed_dict={handle: training_handle})
        _, l = sess.run([g_train_op, all_losses], feed_dict={handle: training_handle})
        return l

    return helper

        
def losses(x, g, d_fake, d_real, args, reuse=False):
    """Add loss nodes to the graph depending on the model type.

    This implements an Improved Wasserstein loss with an additional
    reconstruction loss term in the generator.

    Args:
        x: Tensor, the real (input) images.
        g: Tensor, the fake images (i.e. the output from the generator)
        d_fake: Tensor, the discriminator using the fake images.
        d_real: Tensor, the discriminator using the real images.
        args: Argparse struct.
        reuse: Boolean, whether to reuse the variables in this scope.

    Returns:
        g_loss: Op, the loss op for the generator.
        d_loss: Op, the loss op for the discriminator.

    """
    # generator loss
    _, x_depth = _split(x)
    _, g_depth = _split(g)
    if not reuse:
        hem.histograms(('loss_x_depth', x_depth), ('loss_g_depth', g_depth))
    # rescaled to [0, 1]
    x_depth = (x_depth + 1.0)/2.0
    g_depth = (g_depth + 1.0)/2.0

    if not reuse:
        hem.histograms(('loss_x_depth_rescaled', x_depth), ('loss_g_depth_rescaled', g_depth))

    l_term = 1.0
    rmse_loss = hem.rmse(x_depth, g_depth, name='rmse')
    g_loss = tf.identity(-tf.reduce_mean(d_fake) + l_term * rmse_loss, name='g_loss')

    # discriminator loss
    l_term = 10.0
    with tf.variable_scope('discriminator'):
        gp = tf.identity(_gradient_penalty(x, g, args), 'grad_penalty')
    d_gan_loss = tf.identity(tf.reduce_mean(d_fake) - tf.reduce_mean(d_real), name='d_loss1')
    d_grad_penalty_loss = tf.identity(l_term * gp, name='d_loss2')
    d_loss = tf.identity(d_gan_loss + d_grad_penalty_loss, name='d_loss')

    # track losses in collection node
    if not reuse:
        for l in [g_loss, d_loss, rmse_loss, d_gan_loss, d_grad_penalty_loss]:
            tf.add_to_collection('losses', l)

    return g_loss, d_loss


def _gradient_penalty(x, g, args):
    """Calculate gradient penalty for discriminator.

    Source: `Improved Training of Wasserstein GANs`
    https://arxiv.org/abs/1704.00028

    Args:
        x: Tensor, the real (input) images.
        g: Tensor, the fake (generator) images.
        args: Argparse struct.

    Returns:
        Tensor, the gradient penalty term.
    """
    x = hem.flatten(x)
    g = hem.flatten(g)
    alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0, maxval=1.0)
    interpolates = x + (alpha * (g - x))
    interpolates = tf.reshape(interpolates, (-1, 4, 64, 64))
    d = discriminator(interpolates, args, reuse=True)
    gradients = tf.gradients(d, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
    penalty = tf.reduce_mean((slopes-1.0)**2)
    return penalty


def generator(x, args, reuse=False):
    """Adds generator nodes to the graph.

    This is a typical autoencoder architecture where the input is the image x
    and a nosie vector, and the output is an estimated depth map.

    Args:
        x: Tensor, the input tensor.
        args: Argparse struct.
        reuse: Boolean, whether to reuse the variables in this scope.

    Returns:
        Tensor, the output of the constructed generator.
    """
    e = _encoder(x, args, reuse)
    d = _decoder(e, args, reuse)
    g = tf.concat((x, d), axis=1)
    return g


def _encoder(x, args, reuse=False):
    """Adds encoder nodes to the graph.

    Args:
        x: Tensor, the input tensor/images.
        args: Arparse struct.
        reuse: Boolean, whether to reuse variables.

    Returns:
        Tensor, the encoded latent variables for the given input.
    """
    with arg_scope([hem.conv2d],
                   reuse=reuse,
                   use_batch_norm=args.batch_norm,
                   dropout=args.dropout,
                   activation=hem.lrelu):
        noise = tf.random_uniform((args.batch_size, 64, 64, 1), minval=-1.0, maxval=1.0)
        noise = tf.transpose(noise, [0, 3, 1, 2])
        x = tf.concat([x, noise], axis=1)
        # input = 64x646x4
        x = hem.conv2d(x,   4,  64, 5, 2, name='e_c1', dropout=0, use_batch_norm=False)  # output = 32x32x64
        x = hem.conv2d(x,  64, 128, 5, 2, name='e_c2')  # output = 16x16x128
        x = hem.conv2d(x, 128, 256, 5, 2, name='e_c3')  # output = 8x8x256
        x = hem.conv2d(x, 256, 256, 5, 2, name='e_c4')  # output = 4x4x256
        x = hem.conv2d(x, 256,  96, 1,    name='e_c5')  # output = 4x4x96
        x = hem.conv2d(x,  96,  32, 1,    name='e_c6')  # output = 4x4x32

    return x


def _decoder(x, args, reuse=False):
    """
    Adds decoder nodes to the graph.

    Args:
        x: Tensor, the latent variable tensor from the encoder.
        args: Argparse structure.
        reuse: Boolean, whether to reuse pinned variables (in multi-GPU environment).

    Returns:
        Tensor, a single-channel image representing the predicted depth map.
    """
    e_layers = tf.get_collection('conv_layers')
    
    with arg_scope([hem.dense, hem.conv2d, hem.deconv2d],
                   reuse=reuse,
                   use_batch_norm=args.batch_norm,
                   dropout=args.dropout,
                   activation=tf.nn.relu):
        skip_axis = 1
        skip_m = 2 if args.skip_layers else 1

        # TODO add better support for skip layers in layer ops
        # input = 4x4x32
        x = hem.conv2d(x,    32,  96, 1,    name='d_c1')           # output = 4x4x96    + e_c5
        x = tf.concat((x, e_layers[4]), axis=skip_axis) if args.skip_layers else x
        x = hem.conv2d(x,    96*skip_m, 256, 1,    name='d_c2')    # output = 4x4x256   + e_c4
        x = tf.concat((x, e_layers[3]), axis=skip_axis) if args.skip_layers else x

        x = hem.deconv2d(x, 256*skip_m, 256, 5, 2, name='d_dc1')   # output = 8x8x256   + e_c3


        x = tf.concat((x, e_layers[2]), axis=skip_axis) if args.skip_layers else x
        x = hem.deconv2d(x, 256*skip_m, 128, 5, 2, name='d_dc2')   # output = 16x16x128 + e_c2
        x = tf.concat((x, e_layers[1]), axis=skip_axis) if args.skip_layers else x
        x = hem.deconv2d(x, 128*skip_m,  64, 5, 2, name='d_dc3')   # output = 32x32x64  + e_c1
        x = tf.concat((x, e_layers[0]), axis=skip_axis) if args.skip_layers else x
        x = hem.deconv2d(x,  64*skip_m,   1, 5, 2, name='d_dc4', activation=tf.nn.tanh, dropout=0, use_batch_norm=False)  # output = 64x64x1
    return x



def discriminator(x, args, reuse=False):
    """Adds discriminator nodes to the graph.

    From the input image, successively applies convolutions with
    striding to scale down layer sizes until we get to a single
    output value, representing the discriminator's estimate of fake
    vs real.

    Args:
        x: Tensor, the input (image) tensor. This should be a 4D Tensor with 3 RGB channels† and 1 depth channel.
        args: Argparse struct.
        reuse: Boolean, whether to reuse the variables in this scope.

    Returns:
        Tensor, the discriminator's estimation of whether the inputs are fake or real.
    """
    with arg_scope([hem.conv2d],
                   activation=hem.lrelu,
                   reuse=reuse):
        x = hem.conv2d(x, 4,    64, 5, 2, name='disc_c1')
        x = hem.conv2d(x, 64,  128, 5, 2, name='disc_c2')
        x = hem.conv2d(x, 128, 256, 5, 2, name='disc_c3')
        x = hem.conv2d(x, 256, 512, 1,    name='disc_c4')
        x = hem.conv2d(x, 512, 1,   1,    name='disc_c5', activation=tf.nn.sigmoid)
    return x


def _split(x, reshape_to=None, name=None):
    """Split the 4D tesnor into RGB and Depth components.

    Args:
        x: Tensor, 4D tensor (images + depths)
        name: String, name for this op.

    Returns:
        Tuple, contains two Tensors representing the image and depth, respectively
    """
    with tf.name_scope(name, 'split', [x]):
        rgb = x[:, 0:3, :, :]
        depth = x[:, 3, :, :]
        depth = tf.expand_dims(depth, 1)
    return rgb, depth