import tensorflow as tf
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, L



class VariationalAutoEncoder(Model):
    def __init__(self, x): # optimizer):
        self.latent_size = 200
        self.batch_size = 256
        
        # encoder
        self._encoder_node = self._build_encoder(x)
        # latent
        z_mean, z_stddev = self._build_latent(self._encoder_node)
        samples = tf.random_normal([self.batch_size, self.latent_size], 0, 1, dtype=tf.float32)
        self._latent_node = z_mean + (z_stddev * samples)  # sampling of latent_size gaussians based on z_mean, z_stddev
        # decoder
        self._decoder_node = self._build_decoder(self._latent_node)

        # loss
        generated_loss = -tf.reduce_sum(x * tf.log(1e-8 + self._decoder_node) + (1 - x) * tf.log(1e-8 + (1 - self._decoder_node)))
        latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-8 + tf.square(z_stddev)) - 1) #, 1)
        self._generated_loss_node = tf.reduce_sum(generated_loss)
        self._latent_loss_node = tf.reduce_sum(latent_loss)
        self._loss_node = tf.reduce_mean(generated_loss + latent_loss)
        

        
    def _build_encoder(self, x):
        # layer_sizes = [64, 128, 256, 256, 96, 32] 
        with tf.variable_scope('encoder'):
            x = lrelu(conv2d(x, 1, 64, 5, 2))       ; L(x)
            x = lrelu(conv2d(x, 64, 128, 5, 2))     ; L(x)
            x = lrelu(conv2d(x, 128, 256, 5, 2))    ; L(x)
            x = lrelu(conv2d(x, 256, 256, 5, 2))    ; L(x)
            x = lrelu(conv2d(x, 256, 96, 1))        ; L(x)
            x = lrelu(conv2d(x, 96, 32, 1))         ; L(x)
        return x

    
    def _build_latent(self, x):
        with tf.variable_scope('latent'):
            flat = flatten(x)
            w_mean = dense(flat, 32*4*4, self.latent_size)       ; L(w_mean)
            w_stddev = dense(flat, 32*4*4, self.latent_size)     ; L(w_stddev)
        return w_mean, w_stddev

    
    def _build_decoder(self, x):
        # layer sizes = [96, 256, 256, 128, 64]
        with tf.variable_scope('decoder'):
            x = dense(x, self.latent_size, 32*4*4)
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
            
            x = tf.nn.relu(conv2d(x, 32, 96, 1))         ; L(x)
            x = tf.nn.relu(conv2d(x, 96, 256, 1))        ; L(x)
            x = tf.nn.relu(deconv2d(x, 256, 256, 5, 2))  ; L(x)
            x = tf.nn.relu(deconv2d(x, 256, 128, 5, 2))  ; L(x)
            x = tf.nn.relu(deconv2d(x, 128, 64, 5, 2))   ; L(x)
            x = tf.nn.sigmoid(deconv2d(x, 64, 1, 5, 2))  ; L(x)
        return x

    
            

