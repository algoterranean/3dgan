import tensorflow as tf
import numpy as np
import time
from sys import stdout
from util import print_progress, fold
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, L, M



class VAE(Model):
    def __init__(self, optimizer, x):
        Model.__init__(self)
        self.latent_size = 512
        self.batch_size = 256
        self._optimizer = optimizer
        
        # encoder
        self._encoder = self._build_encoder(x)                   ; M(self._encoder, 'outputs')    
        # latent
        self._latent, self._z_mean, self._z_stddev = self._build_latent(self._encoder) ; M(self._latent, 'outputs')
        # decoder
        self._decoder = self._build_decoder(self._latent)       ; M(self._decoder, 'outputs')
        
        # tf.add_to_collection('epoch_summaries', tf.summary.image(self._encoder.name, self._encoder))
        tf.add_to_collection('epoch_summaries', tf.summary.image('input images', x))
        tf.add_to_collection('epoch_summaries', tf.summary.image('output images', self._decoder))
        
        # loss
        with tf.variable_scope('losses'):
            with tf.variable_scope('generated_loss'):
                generated_loss = -tf.reduce_sum(x * tf.log(1e-8 + self._decoder) + \
                                                    (1 - x) * tf.log(1e-8 + (1 - self._decoder)))
            with tf.variable_scope('latent_loss'):
                latent_loss = 0.5 * tf.reduce_sum(tf.square(self._z_mean) + \
                                                      tf.square(self._z_stddev) - \
                                                      tf.log(1e-8 + tf.square(self._z_stddev)) - 1)
            with tf.variable_scope('loss'):
                self._loss = tf.reduce_mean(generated_loss + latent_loss)     ; M(self._loss, 'losses')

            self._generated_loss = tf.reduce_sum(generated_loss)          ; M(self._generated_loss, 'losses')
            self._latent_loss = tf.reduce_sum(latent_loss)                ; M(self._latent_loss, 'losses')

        with tf.variable_scope('train_op'):
            self._train_op = optimizer.minimize(self._loss)

        # add scalar summaries 
        tf.add_to_collection('batch_summaries', tf.summary.scalar('loss', self._loss))
        tf.add_to_collection('batch_summaries', tf.summary.scalar('generator loss', self._generated_loss))
        tf.add_to_collection('batch_summaries', tf.summary.scalar('latent loss', self._latent_loss))


    def _build_encoder(self, x):
        """Input: 64x64x3. Output: 4x4x32."""
        # layer_sizes = [64, 128, 256, 256, 96, 32] 
        with tf.variable_scope('encoder') as scope:
            with tf.variable_scope('conv1'):
                x = lrelu(conv2d(x, 3, 64, 5, 2))      ; M(x, 'layer')
            with tf.variable_scope('conv2'):
                x = lrelu(conv2d(x, 64, 128, 5, 2))    ; M(x, 'layer')
            with tf.variable_scope('conv3'):
                x = lrelu(conv2d(x, 128, 256, 5, 2))    ; M(x, 'layer')
            with tf.variable_scope('conv4'):
                x = lrelu(conv2d(x, 256, 256, 5, 2))    ; M(x, 'layer')
            with tf.variable_scope('conv5'):
                x = lrelu(conv2d(x, 256, 96, 1))        ; M(x, 'layer')
            with tf.variable_scope('conv6'):
                x = lrelu(conv2d(x, 96, 32, 1))         ; M(x, 'layer')
            tf.identity(x, name='sample')
        return x

    
    def _build_latent(self, x):
        """Input: 4x4x32. Output: 200."""
        with tf.variable_scope('latent'):
            with tf.variable_scope('flatten'):
                flat = flatten(x)
            with tf.variable_scope('mean'):
                z_mean = dense(flat, 32*4*4, self.latent_size)       ; M(z_mean, 'layer')
            with tf.variable_scope('stddev'):
                z_stddev = dense(flat, 32*4*4, self.latent_size)     ; M(z_stddev, 'layer')
            with tf.variable_scope('gaussian'):
                samples = tf.random_normal([self.batch_size, self.latent_size], 0, 1, dtype=tf.float32)    ; M(samples, 'layer')
                z = (z_mean + (z_stddev * samples))                  ; M(z, 'layer')
            tf.identity(z, name='sample')
        return (z, z_mean, z_stddev)

    
    def _build_decoder(self, x):
        """Input: 200. Output: 64x64x3."""
        # layer sizes = [96, 256, 256, 128, 64]
        with tf.variable_scope('decoder'):
            with tf.variable_scope('dense'):
                x = dense(x, self.latent_size, 32*4*4)       ; M(x, 'layer')
            with tf.variable_scope('conv1'):
                x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
                x = tf.nn.relu(conv2d(x, 32, 96, 1))         ; M(x, 'layer')
            with tf.variable_scope('conv2'):
                x = tf.nn.relu(conv2d(x, 96, 256, 1))        ; M(x, 'layer')
            with tf.variable_scope('deconv1'):
                x = tf.nn.relu(deconv2d(x, 256, 256, 5, 2))  ; M(x, 'layer')
            with tf.variable_scope('deconv2'):
                x = tf.nn.relu(deconv2d(x, 256, 128, 5, 2))  ; M(x, 'layer')
            with tf.variable_scope('deconv3'):
                x = tf.nn.relu(deconv2d(x, 128, 64, 5, 2))   ; M(x, 'layer')
            with tf.variable_scope('deconv4'):
                x = tf.nn.sigmoid(deconv2d(x, 64, 3, 5, 2))  ; M(x, 'layer')
            tf.identity(x, name='sample')
        return x


    def train(self, epoch, x_input, data, batch_size, tb_writer, batch_summary_nodes, epoch_summary_nodes, display=True):
        epoch_start_time = time.time()
        n_batches = int(data.train.num_examples/batch_size)
        summary_freq = int(n_batches / 20)  # 10 datapoints per epoch
        
        for i in range(n_batches):
            xs, ys = data.train.next_batch(batch_size)
            # _, l, gl, ll, summary = self._sess.run([self._train_op, self.loss, self.generated_loss, self.latent_loss, batch_summary_nodes], feed_dict={x_input: xs})
            _, l, gl, ll = self._sess.run([self._train_op, self.loss, self.generated_loss, self.latent_loss], feed_dict={x_input: xs})            
            print_progress(epoch, batch_size*(i+1), data.train.num_examples, epoch_start_time, {'loss': l, 'gen_loss': gl, 'latent_loss': ll})

            # batch summary statistics
            if i % summary_freq == 0:
                summary = self._sess.run(batch_summary_nodes, feed_dict={x_input:xs})
                tb_writer.add_summary(summary, batch_size*(i+1) + (epoch-1)*n_batches*batch_size)
                

        # perform validation
        results = fold(self._sess, x_input, [self.loss], data.validation, batch_size, int(data.validation.num_examples/batch_size))
        stdout.write(', validation: {:.4f}\r\n'.format(results[0]))

        # epoch summaries
        summary = self._sess.run(epoch_summary_nodes, feed_dict={x_input: xs})
        tb_writer.add_summary(summary, batch_size*(i+1) + (epoch-1)*n_batches*batch_size)
        

    
