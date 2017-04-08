import tensorflow as tf

def _downconv_layer(x, num_filters, name):
    in_shape = x.get_shape().as_list()
    # weights
    K = tf.Variable(tf.truncated_normal([3, 3, in_shape[3], num_filters], stddev=0.1))
    # bias
    b = tf.Variable(tf.truncated_normal([num_filters]))
    # layer
    l = tf.add(tf.nn.conv2d(x, K, strides=[1, 1, 1, 1], padding='SAME'), b)
    l2 = tf.nn.relu(l)
    l = tf.nn.max_pool(l2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    return l


def _upconv_layer(x, num_filters, name):
    in_shape = x.get_shape().as_list()
    # weights
    K = tf.Variable(tf.truncated_normal([3, 3, num_filters, in_shape[3]], stddev=0.1))
    # bias
    b = tf.Variable(tf.truncated_normal([num_filters]))
    # layer
    in_shape = tf.shape(x)
    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
    l = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME'), b)
    l = tf.add(l, tf.Variable(tf.zeros([K.get_shape().as_list()[2]])))
    l = tf.nn.relu(l, name=name)
    return l
    
# TODO: build decoder from same weights as encoder
def simple_cnn(x, layer_sizes):
    # input
    orig_shape = x.get_shape().as_list()
    summary_nodes = []    

    # encoder
    with tf.variable_scope('encoder'):
        for out_size in layer_sizes:
            x = _downconv_layer(x, out_size, 'Layer.Encoder.{}'.format(out_size))
            tf.add_to_collection('layers', x)
            summary_nodes.append(tf.summary.histogram('Encoder {}'.format(out_size), x))


    # decoder
    with tf.variable_scope('decoder'):
        for out_size in layer_sizes[1::-1]:
            x = _upconv_layer(x, out_size, 'Layer.Decoder.{}'.format(out_size))
            tf.add_to_collection('layers', x)
            summary_nodes.append(tf.summary.histogram('Decoder {}'.format(out_size),x))

        # output
        x = _upconv_layer(x, orig_shape[3], 'Layer.Output')
        tf.add_to_collection('layers', x)
        summary_nodes.append(tf.summary.histogram('Output', x))
        
    return x, tf.summary.merge(summary_nodes)



