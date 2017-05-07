import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import time
from sys import stdout
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, M
from util import print_progress, fold




class GAN(Model):
    def __init__(self, x, batch_size):
        Model.__init__(self)
        self.latent_size = 512
        self.batch_size = batch_size
        # self._optimizer = optimizer
        
        self._x_input = x
        # latent
        self._latent = self._build_latent()
        # generator
        self._decoder = self._build_decoder(self._latent)

        
        with tf.variable_scope('discriminator') as scope:
            # discriminator (real)
            self._disc_real, self._disc_logits_real = self._build_discriminator(x)
            # discriminator (fake)
            scope.reuse_variables()
            self._disc_fake, self._disc_logits_fake = self._build_discriminator(self._decoder)

        tf.add_to_collection('epoch_summaries', tf.summary.image('input images', x))
        tf.add_to_collection('epoch_summaries', tf.summary.image('generator images', self._decoder))
        
        
        # loss
        with tf.variable_scope('losses'):
            # need to maximize this, but TF only does minimization, so we use negative
            
            self._loss_d = tf.reduce_mean(-tf.log(self._disc_real + 1e-8) - \
                                              tf.log(1 - self._disc_fake + 1e-8))   ; M(self._loss_d, 'losses')
            self._loss_g = tf.reduce_mean(-tf.log(self._disc_fake + 1e-8))          ; M(self._loss_g, 'losses')

        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/decoder')
        print('discriminator vars\n', '='*40)
        for d in d_vars:
            print(d.name)

        print('generator vars\n', '='*40)
        for g in g_vars:
            print(g.name)
            
        
        # optimizers
        with tf.variable_scope('train_op'):
            # self._disc_optimizer = tf.train.GradientDescentOptimizer().minimize(self._loss_d, var_list=d_vars)
            self._disc_optimizer = tf.train.AdamOptimizer(0.0002, 0.5).minimize(self._loss_d, var_list=d_vars)            
            self._gen_optimizer = tf.train.AdamOptimizer(0.0002, 0.5).minimize(self._loss_g, var_list=g_vars)
            # self._disc_optimizer = optimizer.minimize(self._loss_d, var_list=d_vars)
            # self._gen_optimizer = optimizer.minimize(self._loss_g, var_list=g_vars)


        tf.add_to_collection('batch_summaries', tf.summary.scalar('loss_d', self._loss_d)) #, var_list='hi'))
        tf.add_to_collection('batch_summaries', tf.summary.scalar('loss_g', self._loss_g))

        
        
    def _build_latent(self):
        """Input: 4x4x32. Output: 200."""
        with tf.variable_scope('latent') as scope:
            z = tf.random_normal([self.batch_size, self.latent_size], 0, 1)
            # z = tf.placeholder(tf.float32, [None, self.latent_size])    ; M(z, 'layer')
            tf.identity(z, name='sample')
        return z

    
    def _build_decoder(self, x):
        """Input: 200. Output: 64x64x3."""
        with tf.variable_scope('decoder') as scope:
            # layer sizes = [96, 256, 256, 128, 64]
            with tf.variable_scope('fit_and_reshape'):
                x = tf.nn.relu(batch_norm(dense(x, self.latent_size, 32*4*4, name='d1')))
                x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
            with tf.variable_scope('conv1'):
                x = tf.nn.relu(batch_norm(conv2d(x, 32, 96, 1, name='c1')))            ; M(x, 'layer')
            with tf.variable_scope('conv2'):                
                x = tf.nn.relu(batch_norm(conv2d(x, 96, 256, 1, name='c2')))           ; M(x, 'layer')
            with tf.variable_scope('deconv1'):                
                x = tf.nn.relu(batch_norm(deconv2d(x, 256, 256, 5, 2,name='dc1')))     ; M(x, 'layer')
            with tf.variable_scope('deconv2'):                
                x = tf.nn.relu(batch_norm(deconv2d(x, 256, 128, 5, 2,name='dc2')))     ; M(x, 'layer')
            with tf.variable_scope('deconv3'):                
                x = tf.nn.relu(batch_norm(deconv2d(x, 128, 64, 5, 2, name='dc3')))      ; M(x, 'layer')
            with tf.variable_scope('deconv4'):
                x = tf.nn.relu(batch_norm(deconv2d(x, 64, 3, 5, 2, name='dc4')))       ; M(x, 'layer')
            with tf.variable_scope('output'):
                x = tf.tanh(x, name='out')                                        ; M(x, 'layer')
                # x = tf.nn.sigmoid(deconv2d(x, 64, 3, 5, 2, name='dc4'))     ; M(x, 'layer')
            tf.identity(x, name='sample')
        return x
    

    def _build_discriminator(self, x):
        # with tf.variable_scope('discriminator') as scope:
            # layer sizes = [64, 128, 256, 256, 96]
        with tf.variable_scope('conv1'):
            x = lrelu(conv2d(x, 3, 64, 5, 2, name='c1'))                        ; M(x, 'layer')
        with tf.variable_scope('conv2'):
            x = lrelu(batch_norm(conv2d(x, 64, 128, 5, 2, name='c2')))          ; M(x, 'layer')
        with tf.variable_scope('conv3'):
            x = lrelu(batch_norm(conv2d(x, 128, 256, 5, 2, name='c3')))         ; M(x, 'layer')
        with tf.variable_scope('conv4'):
            x = lrelu(batch_norm(conv2d(x, 256, 256, 5, 2, name='c4')))         ; M(x, 'layer')
        with tf.variable_scope('conv5'):
            x = lrelu(batch_norm(conv2d(x, 256, 96, 32, 1, name='c5')))         ; M(x, 'layer')
        with tf.variable_scope('conv6'):
            x = lrelu(batch_norm(conv2d(x, 96, 32, 1, name='c6')))              ; M(x, 'layer')
        with tf.variable_scope('flatten'):
            logits = flatten(x, name='flat1')
        with tf.variable_scope('logits'):
            out = tf.nn.sigmoid(logits, name='sig1')                             ; M(out, 'layer')
        tf.identity(out, name='sample')
        return out, logits

    # The ideal minimum loss for the discriminator is 0.5 - this is where the generated images are indistinguishable from the real images from the perspective of the discriminator.

    def train(self, epoch, x_input, data, batch_size, tb_writer, batch_summary_nodes, epoch_summary_nodes, display=True):
        epoch_start_time = time.time()
        n_batches = int(data.train.num_examples/batch_size)
        summary_freq = int(n_batches / 20)
        
        for i in range(n_batches):
            # train discriminator
            xs, ys = data.train.next_batch(batch_size)
            # zs = np.random.normal(0, 1, (batch_size, 512))
            _, _, dl, gl = self._sess.run([self._disc_optimizer, self._gen_optimizer, self._loss_d, self._loss_g], feed_dict={self._x_input: xs})
            
            # _, dl = self._sess.run([self._disc_optimizer, self._loss_d], feed_dict={self._x_input: xs})
            # _, gl = self._sess.run([self._gen_optimizer, self._loss_g], feed_dict={self._x_input: xs})
            
            # _, dl = self._sess.run([self._disc_optimizer, self._loss_d], feed_dict={self._x_input: xs, self._latent: zs})
            # # train generator
            # zs = np.random.normal(0, 1, (256, 512))
            # _, gl = self._sess.run([self._gen_optimizer, self._loss_g], feed_dict={self._latent: zs, self._x_input: xs})
            print_progress(epoch, batch_size*(i+1), data.train.num_examples, epoch_start_time, {'disc loss': dl, 'gen loss': gl})

            if i % summary_freq == 0:
                summary = self._sess.run(batch_summary_nodes, feed_dict={self._x_input: xs}) #, self._latent: zs})
                tb_writer.add_summary(summary, batch_size*(i+1) + (epoch-1)*n_batches*batch_size)
                
            
            # _, l, summary = self._sess.run([self._train_op, self.loss, summary_nodes], feed_dict={x_input: xs})
            # print_progress(epoch, batch_size*(i+1), data.train.num_examples, epoch_start_time, {'loss': l})
            # tb_writer.add_summary(summary, i * batch_size + (epoch-1) * n_batches * batch_size)
            # tb_writer.add_summary(summary2, i * batch_size + (epoch-1) * n_batches * batch_size)

        # epoch summaries
        summary = self._sess.run(epoch_summary_nodes, feed_dict={self._x_input: xs}) #, self._latent: zs})
        tb_writer.add_summary(summary, batch_size*(i+1) + (epoch-1)*n_batches*batch_size)

        # # perform validation
        # results = fold(self._sess, x_input, [self.loss], data.validation, batch_size, int(data.validation.num_examples/batch_size))
        # stdout.write(', validation: {:.4f}\r\n'.format(results[0]))
        stdout.write('\r\n')



    
