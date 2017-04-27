import tensorflow as tf
from .layers import downconv_layer, upconv_layer
from .model import Model

    
# TODO: build decoder from same weights as encoder

class SimpleCNN(Model):
    def __init__(self, x, layer_sizes):
        self.layer_sizes = layer_sizes
        Model.__init__(self, x)

    def _build_graph(self, x):
        orig_shape = x.get_shape().as_list()

        # encoder
        with tf.variable_scope('encoder'):
            for out_size in layer_sizes:
                x = downconv_layer(x, out_size, max_pool=2, name='Layer.Encoder.{}'.format(out_size))
                tf.add_to_collection('layers', x)
                self.summary_nodes.append(tf.summary.histogram('Encoder {}'.format(out_size), x))

        # decoder
        with tf.variable_scope('decoder'):
            for out_size in layer_sizes[1::-1]:
                x = upconv_layer(x, out_size, name='Layer.Decoder.{}'.format(out_size))
                tf.add_to_collection('layers', x)
                self.summary_nodes.append(tf.summary.histogram('Decoder {}'.format(out_size),x))
            # output
            x = upconv_layer(x, orig_shape[3], name='Layer.Output')
            tf.add_to_collection('layers', x)
            self.summary_nodes.append(tf.summary.histogram('Output', x))
            
        self.output = x
        



