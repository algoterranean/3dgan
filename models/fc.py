import tensorflow as tf


def _fc_layer(x, x_size, y_size):
    W = tf.Variable(tf.random_normal([x_size, y_size]))
    b = tf.Variable(tf.random_normal([y_size]))
    l = tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))
    return l



def simple_fc(x, layer_sizes):
    orig_shape = list(x.get_shape())
    summary_nodes = []

    with tf.variable_scope('input'):
        # flatten        
        x = tf.contrib.layers.flatten(x)
        flattened_size = int(list(x.get_shape())[1])
        print('input layer:', x, x.get_shape())
        summary_nodes.append(tf.summary.histogram('Input', x))
        
    with tf.variable_scope('encoder'):
        # encoder
        for size in layer_sizes:
            s = int(list(x.get_shape())[1])
            x = _fc_layer(x, s, size)
            summary_nodes.append(tf.summary.histogram('Encoder {}'.format(size), x))
            print('encoder:', x, x.get_shape())

    with tf.variable_scope('decoder'):
        # decoder
        # layer_sizes = list(reversed(layer_sizes[1:]))
        # print('layer_sizes:', layer_sizes, layer_sizes[1::-1][1:])
        for size in layer_sizes[1::-1][1:]:
            s = int(list(x.get_shape())[1])
            x = _fc_layer(x, s, size)
            summary_nodes.append(tf.summary.histogram('Decoder {}'.format(size), x))
            print('decoder:', x, x.get_shape())

        x = _fc_layer(x, int(list(x.get_shape())[1]), flattened_size)
        summary_nodes.append(tf.summary.histogram('Output', x))
        print('final layer:', x, x.get_shape())

    with tf.variable_scope('output'):
        # unflatten
        l = list(orig_shape)[1:]
        l = [-1, int(l[0]), int(l[1]), int(l[2])]
        print('reshape:', l)
        x = tf.reshape(x, l)

    return x, tf.summary.merge(summary_nodes)
