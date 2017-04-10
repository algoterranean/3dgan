import tensorflow as tf
from .layers import downconv_layer, upconv_layer


    
# TODO: build decoder from same weights as encoder


def simple_cnn(x, layer_sizes):
    # input
    orig_shape = x.get_shape().as_list()
    summary_nodes = []    

    # encoder
    with tf.variable_scope('encoder'):
        for out_size in layer_sizes:
            x = downconv_layer(x, out_size, max_pool=2, name='Layer.Encoder.{}'.format(out_size))
            tf.add_to_collection('layers', x)
            summary_nodes.append(tf.summary.histogram('Encoder {}'.format(out_size), x))


    # decoder
    with tf.variable_scope('decoder'):
        for out_size in layer_sizes[1::-1]:
            x = upconv_layer(x, out_size, name='Layer.Decoder.{}'.format(out_size))
            tf.add_to_collection('layers', x)
            summary_nodes.append(tf.summary.histogram('Decoder {}'.format(out_size),x))

        # output
        x = upconv_layer(x, orig_shape[3], name='Layer.Output')
        tf.add_to_collection('layers', x)
        summary_nodes.append(tf.summary.histogram('Output', x))
        
    return x, tf.summary.merge(summary_nodes)



