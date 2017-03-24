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




def _chen_downconv_layer(x, num_filters, ksize, stride, name):
    in_shape = x.get_shape().as_list()
    K = tf.Variable(tf.truncated_normal([ksize, ksize, in_shape[3], num_filters], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([num_filters]))
    l = tf.add(tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME'), b)
    l = tf.contrib.layers.batch_norm(l)
    l = tf.nn.relu(l, name=name)
    tf.add_to_collection('layers', l)
    return l

def _chen_upconv_layer(x, num_filters, ksize, stride, name):
    in_shape = x.get_shape().as_list()
    K = tf.Variable(tf.truncated_normal([ksize, ksize, num_filters, in_shape[3]], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([num_filters]))
    in_shape = tf.shape(x)
    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
    l = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, stride, stride, 1], padding='SAME'), b)
    l = tf.add(l, tf.Variable(tf.zeros([K.get_shape().as_list()[2]])))
    l = tf.nn.relu(l, name=name)
    tf.add_to_collection('layers', l)
    return l



def chen_cnn(x):
    orig_shape = x.get_shape().as_list()
    summary_nodes = []

    # def helper(x, name):
    #     tf.add_to_collection('layers', x)
    #     sumary_

    # encoder
    with tf.variable_scope('encoder'):
        x = _chen_downconv_layer(x, 64, 5, 2, 'Layer.Encoder.64')
        summary_nodes.append(tf.summary.histogram('Encoder 64', x))
        x = _chen_downconv_layer(x, 128, 5, 2, 'Layer.Encoder.128')
        summary_nodes.append(tf.summary.histogram('Encoder 128', x))
        x = _chen_downconv_layer(x, 256, 5, 2, 'Layer.Encoder.256')
        summary_nodes.append(tf.summary.histogram('Encoder 256', x))
        x = _chen_downconv_layer(x, 256, 5, 2, 'Layer.Encoder.256x2')
        summary_nodes.append(tf.summary.histogram('Encoder 256x2', x))
        x = _chen_downconv_layer(x, 96, 1, 1, 'Layer.Encoder.96')
        summary_nodes.append(tf.summary.histogram('Encoder 96', x))        
        x = _chen_downconv_layer(x, 32, 1, 1, 'Layer.Encoder.32')
        summary_nodes.append(tf.summary.histogram('Encoder 32', x))

    # decoder
    with tf.variable_scope('decoder'):
        x = _chen_downconv_layer(x, 96, 1, 1, 'Layer.Decoder.96')
        summary_nodes.append(tf.summary.histogram('Decoder 96', x))        
        x = _chen_downconv_layer(x, 256, 1, 1, 'Layer.Decoder.256')
        summary_nodes.append(tf.summary.histogram('Decoder 256', x))                
        x = _chen_upconv_layer(x, 256, 5, 2, 'Layer.Decoder.256x2')
        summary_nodes.append(tf.summary.histogram('Decoder 256x2', x))                
        x = _chen_upconv_layer(x, 128, 5, 2, 'Layer.Decoder.128')
        summary_nodes.append(tf.summary.histogram('Decoder 128', x))                
        x = _chen_upconv_layer(x, 64, 5, 2, 'Layer.Decoder.64')
        summary_nodes.append(tf.summary.histogram('Decoder 64', x))                
        # output
        x = _chen_upconv_layer(x, 3, 5, 2, 'Layer.Decoder.3')
        summary_nodes.append(tf.summary.histogram('Decoder 3', x))                

    return x, tf.summary.merge(summary_nodes)

