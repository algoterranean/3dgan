import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from util import *
from models.model import Model
from models.ops import dense, conv2d, deconv2d, lrelu, flatten, montage_summary



TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
SUMMARIES = tf.GraphKeys.SUMMARIES
UPDATE_OPS = tf.GraphKeys.UPDATE_OPS


class Test(Model):
    def __init__(self, x, args):
        # with tf.variable_scope(tf.get_variable_scope()):        
        g_opt, d_opt = init_optimizer(args), init_optimizer(args)
        global_step = tf.train.get_global_step()

        # real_data_int = tf.placeholder(tf.int32, shape=[batch_size, output_dim])
        # real_data = 2 * ((tf.cast(real_data_int, tf.float32)/255.0)-0.5)

        x = flatten(x)
        tf.summary.histogram('real', x)
        # rescale [0,1] to [-1,1]        
        x = 2 * (x- 0.5)        

        

        g, g_scope = self.generator(args)
        d_real, d_scope = self.discriminator(x, args)
        d_fake, d_scope = self.discriminator(g, args, reuse=True)



        print('SCOPE', g_scope, d_scope)
        # print('STUFF', tf.GraphKeys.TRAINABLE_VARIABLES)
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') #g_scope)
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator') #d_scope)
        
        g_cost = -tf.reduce_mean(d_fake)
        d_cost = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)

        l_term = 10.0
        alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0.0, maxval=1.0)
        differences = g - x
        interpolates = x + (alpha * differences)
        d_interpolates, _ = self.discriminator(interpolates, args, reuse=True)
        gradients = tf.gradients(d_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients))) #, reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
        d_cost += l_term * gradient_penalty

        tf.add_to_collection('losses', g_cost)
        tf.add_to_collection('losses', d_cost)

        
        batchnorm_updates = tf.get_collection(UPDATE_OPS, 'generator')
        g_grads = g_opt.compute_gradients(g_cost, var_list=g_params)
        d_grads = d_opt.compute_gradients(d_cost, var_list=d_params)
        with tf.control_dependencies(batchnorm_updates):
            self.g_train_op = g_opt.apply_gradients(g_grads, global_step=global_step)
            self.d_train_op = d_opt.apply_gradients(d_grads, global_step=global_step)
            # self.g_train_op = g_opt.minimize(g_cost, var_list=g_params, global_step=global_step)
        # self.d_train_op = d_opt.minimize(d_cost, var_list=d_params, global_step=global_step)

        # rescale images from [-1,1] to [0,1]
        real_images = (x[0:args.examples] + 1.0) / 2
        fake_samples = (g[0:args.examples] + 1.0) / 2
        # summaries
        montage_summary(tf.reshape(real_images, [-1, 32, 32, 3]), 8, 8, 'inputs')
        montage_summary(tf.reshape(fake_samples, [-1, 32, 32, 3]), 8, 8, 'fake')

        tf.summary.histogram('fakes', g)
        for grad, var in g_grads:
            tf.summary.histogram(var.name + '/gradient', grad)
        for grad, var in d_grads:
            tf.summary.histogram(var.name + '/gradient', grad)

        tf.summary.scalar('d_loss', d_cost)
        tf.summary.scalar('g_loss', g_cost)
        
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary_op = tf.summary.merge(summaries)

        # gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        # disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
        

    def generator(self, args):
        output_dim = 32*32*3
        with tf.variable_scope('generator') as scope:
            z = tf.random_normal([args.batch_size, args.latent_size])
            y = tf.reshape(tf.nn.relu(batch_norm(dense(z, args.latent_size, 4*4*4*args.latent_size, name='fc1'))), [-1, 4, 4, 4*args.latent_size])
            y = tf.nn.relu(batch_norm(deconv2d(y, 4*args.latent_size, 2*args.latent_size, 5, 2, name='dc1')))
            y = tf.nn.relu(batch_norm(deconv2d(y, 2*args.latent_size, args.latent_size, 5, 2, name='dc2')))
            y = tf.tanh(batch_norm(deconv2d(y, args.latent_size, 3, 5, 2, name='dc3')))
            y = tf.reshape(y, [-1, output_dim])
        return y, scope

    
    def discriminator(self, x, args, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            x = tf.reshape(x, [-1, 32, 32, 3])
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

    


# z_size = 128
# l_term = 10.0
# critic_iters = 5
# batch_size = 64
# output_dim = 32*32*3


# def get_images(batch_size):
#     all_data = []
#     for x in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
#         f = open(x, 'rb')
#         dict = pickle.load(f, encoding='bytes')
#         f.close()
#         all_data.append(dict[b'data'])
#     images = np.concatenate(all_data, axis=0)

#     def get_epoch():
#       np.random.shuffle(images)
#       for i in xrange(len(images) / batch_size):
#         yield np.copy(images[i*batch_size:(i+1)*batch_size])
#     return get_epoch
  

# def generator(batch_size, z_size, output_dim):
#     with tf.variable_scope('generator') as scope:
#       z = tf.random_normal([batch_size, z_size])
#       y = tf.reshape(tf.nn.relu(batch_norm(dense(z, z_size, 4*4*4*z_size, name='fc1'))), [-1, 4, 4, 4*z_size])
#       y = tf.nn.relu(batch_norm(deconv2d(y, 4*z_size, 2*z_size, 5, 2, name='dc1')))
#       y = tf.nn.relu(batch_norm(deconv2d(y, 2*z_size, z_size, 5, 2, name='dc2')))
#       y = tf.tanh(batch_norm(devonc2d(y, z_size, 3, 5, 2, name='dc3')))
#       y = tf.reshape(y, [-1, output_dim])
#     return y, scope

  
# def discriminator(x, z_size, reuse=False):
#     with tf.variable_scope('discriminator') as scope:
#         x = tf.reshape(x, [-1, 32, 32, 3])
#         x = lrelu(conv2d(x, 3, z_size, 5, 2, reuse=reuse, name='c1'))
#         x = lrelu(conv2d(x, z_size, z_size*2, 5, 2, reuse=reuse, name='c2'))
#         x = lrelu(conv2d(x, z_size*2, z_size*4, 5, 2, reuse=reuse, name='c3'))
#         x = tf.reshape(x, [-1, 4*4*4*z_size])
#         x = dense(x, 4*4*4*z_size, 1, reuse=reuse, name='fc2')
#         x = tf.reshape(x, [-1])
#     return x, scope


# given: images tensor for cifar
# shape = [batch_size, output_dim]

# real_data_int = tf.placeholder(tf.int32, shape=[batch_size, output_dim])
# real_data = 2 * ((tf.cast(real_data_int, tf.float32)/255.0)-0.5)

# fake_data, gen_scope = generator(batch_size, z_size, output_dim)
# disc_real, disc_scope = discriminator(real_data)
# disc_fake, disc_scope = discriminator(fake_data, reuse=True)

# gen_params = tf.get_collection(tf.TRAINABLE_VARIABLES, scope=gen_scope)
# disc_params = tf.get_collection(tf.TRAINABLE_VARIABLES, scope=disc_scope)

# gen_cost = -tf.reduce_mean(disc_fake)
# disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
# differences = fake_data - real_data
# interpolates = real_data + (alpha * differences)
# gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
# slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
# gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
# disc_cost += l_term * gradient_penalty 

# gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
# disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
    

# train_data_gen = get_images(batch_size)
