import tensorflow as tf
from .layers import downconv_layer, upconv_layer
from .model import Model
from .ops import dense, conv2d, deconv2d, lrelu, flatten, L

    
# TODO: build decoder from same weights as encoder



class SimpleCNN(Model):
    def __init__(self, optimizer, x, layer_sizes):
        Model.__init__(self)
        self.layer_sizes = layer_sizes
        orig_shape = x.get_shape().as_list()
        
        self._encoder = self._build_encoder(x)
        self._decoder = self._build_decoder(self._encoder, orig_shape[3])
        self._loss = tf.reduce_mean(tf.abs(x - self._decoder))
        self._optimizer = optimizer        
        self._train_op = optimizer.minimize(self._loss)

        

    def _build_encoder(self, x):
        with tf.variable_scope('encoder'):
            for out_size in self.layer_sizes:
                x = downconv_layer(x, out_size, max_pool=2, name='Layer.Encoder.{}'.format(out_size))     ; L(x)
            tf.identity(x, name='sample')
        return x


    def _build_decoder(self, x, out_filters):
        with tf.variable_scope('decoder'):
            for out_size in self.layer_sizes[1::-1]:
                x = upconv_layer(x, out_size, name='Layer.Decoder.{}'.format(out_size))                   ; L(x)
            x = upconv_layer(x, out_filters, name='Layer.Output')                                         ; L(x)
            tf.identity(x, name='sample')
        return x

