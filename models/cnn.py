import tensorflow as tf
from .layers import downconv_layer, upconv_layer
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten
from util import average_gradients, init_optimizer

    
# TODO: build decoder from same weights as encoder



class CNN(Model):
    def __init__(self, x, args):
        with tf.device('/cpu:0'):
            opt = init_optimizer(args)
            summaries = []
            tower_grads = []

            with tf.variable_scope('model') as scope:
                for gpu_id in range(args.n_gpus):
                    with tf.device(self.variables_on_cpu(gpu_id)):
                        with tf.name_scope('tower_{}'.format(gpu_id)) as scope:
                            encoder, latent, decoder = self.build_model(x, args, gpu_id)

                            losses = tf.get_collection('losses', scope)
                            for l in losses:
                                tf.summary.scalar(self.tensor_name(l), l)
                            loss = losses[0]

                            tf.get_variable_scope().reuse_variables()
                            s = tf.expand_dims(tf.concat(tf.unstack(decoder, num=args.batch_size, axis=0)[0:10], axis=1), axis=0)
                            tf.summary.image('examples', s)

                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
                            with tf.variable_scope('compute_gradients'):
                                grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
                            
            # back on the CPU
            with tf.variable_scope('training'):
                with tf.variable_scope('average_gradients'):
                    grads = average_gradients(tower_grads)
                with tf.variable_scope('apply_gradients'):
                    apply_grad_op = opt.apply_gradients(grads, global_step=tf.train.get_global_step())
                with tf.variable_scope('train_op'):
                    self.train_op = tf.group(apply_grad_op)

            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            self.summary_op = tf.summary.merge(summaries)



    def build_encoder(self, x, reuse=False):
        """Input: 64x64x3. Output: 4x4x32."""
        # layer_sizes = [64, 128, 256, 256, 96, 32] 
        with tf.variable_scope('conv1'):
            x = lrelu(conv2d(x, 3, 64, 5, 2, reuse=reuse, name='c1'))
            self.activation_summary(x)
            
        with tf.variable_scope('conv2'):
            x = lrelu(conv2d(x, 64, 128, 5, 2, reuse=reuse, name='c2'))
            self.activation_summary(x)
            
        with tf.variable_scope('conv3'):
            x = lrelu(conv2d(x, 128, 256, 5, 2, reuse=reuse, name='c3'))
            self.activation_summary(x)
            
        with tf.variable_scope('conv4'):
            x = lrelu(conv2d(x, 256, 256, 5, 2, reuse=reuse, name='c4'))
            self.activation_summary(x)
            
        with tf.variable_scope('conv5'):
            x = lrelu(conv2d(x, 256, 96, 1, reuse=reuse, name='c5'))
            self.activation_summary(x)
            
        with tf.variable_scope('conv6'):
            x = lrelu(conv2d(x, 96, 32, 1, reuse=reuse, name='c6'))
            self.activation_summary(x)
            
        tf.identity(x, name='sample')
        return x

    def build_decoder(self, x, latent_size, reuse=False):
        """Input: 200. Output: 64x64x3."""
        # layer sizes = [96, 256, 256, 128, 64]
        with tf.variable_scope('dense'):
            x = dense(x, latent_size, 32*4*4, reuse=reuse, name='d1')
            self.activation_summary(x)
            
        with tf.variable_scope('conv1'):
            x = tf.reshape(x, [-1, 4, 4, 32]) # un-flatten
            x = tf.nn.relu(conv2d(x, 32, 96, 1, reuse=reuse, name='c1'))
            self.activation_summary(x)
            
        with tf.variable_scope('conv2'):
            x = tf.nn.relu(conv2d(x, 96, 256, 1, reuse=reuse, name='c2'))
            self.activation_summary(x)
            
        with tf.variable_scope('deconv1'):
            x = tf.nn.relu(deconv2d(x, 256, 256, 5, 2, reuse=reuse, name='dc1'))
            self.activation_summary(x)
            
        with tf.variable_scope('deconv2'):
            x = tf.nn.relu(deconv2d(x, 256, 128, 5, 2, reuse=reuse, name='dc2'))
            self.activation_summary(x)
            
        with tf.variable_scope('deconv3'):
            x = tf.nn.relu(deconv2d(x, 128, 64, 5, 2, reuse=reuse, name='dc3'))
            self.activation_summary(x)
            
        with tf.variable_scope('deconv4'):
            x = tf.nn.sigmoid(deconv2d(x, 64, 3, 5, 2, reuse=reuse, name='dc4'))
            self.activation_summary(x)
            
        tf.identity(x, name='sample')
        return x


    def build_latent(self, x, latent_size, reuse=False):
        with tf.variable_scope('flatten'):
            flat = flatten(x)
        with tf.variable_scope('dense'):
            x = dense(flat, 32*4*4, latent_size, reuse=reuse, name='d1')
            self.activation_summary(x)
        return x
    

    def build_model(self, x, args, gpu_id):
        with tf.variable_scope('sliced_input'):
            sliced_input = x[gpu_id * args.batch_size:(gpu_id+1)*args.batch_size,:]
            s = tf.expand_dims(tf.concat(tf.unstack(sliced_input, num=args.batch_size, axis=0)[0:10], axis=1), axis=0)
            tf.summary.image('inputs', s)

        with tf.variable_scope('encoder'):
            encoder = self.build_encoder(sliced_input, (gpu_id > 0))

        with tf.variable_scope('latent'):
            latent = self.build_latent(encoder, args.latent_size, (gpu_id > 0))

        with tf.variable_scope('decoder'):
            decoder = self.build_decoder(latent, args.latent_size, (gpu_id > 0))
        
        with tf.variable_scope('losses'):
            loss = tf.reduce_mean(tf.abs(sliced_input - decoder), name='loss')
        tf.add_to_collection('losses', loss)

        return (encoder, latent, decoder)

            
            


# class SimpleCNN(Model):
#     def __init__(self, optimizer, x, layer_sizes):
#         Model.__init__(self)
#         self.layer_sizes = layer_sizes
#         orig_shape = x.get_shape().as_list()
        
#         self._encoder = self._build_encoder(x)
#         self._decoder = self._build_decoder(self._encoder, orig_shape[3])
#         self._loss = tf.reduce_mean(tf.abs(x - self._decoder))
#         self._optimizer = optimizer        
#         self._train_op = optimizer.minimize(self._loss)

        

#     def _build_encoder(self, x):
#         with tf.variable_scope('encoder'):
#             for out_size in self.layer_sizes:
#                 x = downconv_layer(x, out_size, max_pool=2, name='Layer.Encoder.{}'.format(out_size))     ; L(x)
#             tf.identity(x, name='sample')
#         return x


#     def _build_decoder(self, x, out_filters):
#         with tf.variable_scope('decoder'):
#             for out_size in self.layer_sizes[1::-1]:
#                 x = upconv_layer(x, out_size, name='Layer.Decoder.{}'.format(out_size))                   ; L(x)
#             x = upconv_layer(x, out_filters, name='Layer.Output')                                         ; L(x)
#             tf.identity(x, name='sample')
#         return x

