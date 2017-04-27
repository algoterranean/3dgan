import tensorflow as tf

def _shared_cnn_downconv_layer(x, num_filters, ksize, stride, name, weights=None):
    in_shape = x.get_shape().as_list()
    if weights is None:
        K = tf.Variable(tf.truncated_normal([ksize, ksize, in_shape[3], num_filters], stddev=0.1))
    else:
        # K = weights
        K = tf.transpose(weights, [0, 1, 3, 2])
    b = tf.Variable(tf.truncated_normal([num_filters]))
    l = tf.add(tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME'), b)
    l = tf.contrib.layers.batch_norm(l)
    l = tf.nn.relu(l, name=name)
    tf.add_to_collection('layers', l)
    return l, K


def _shared_cnn_upconv_layer(x, weights, num_filters, ksize, stride, name):
    in_shape = x.get_shape().as_list()
    if weights is None:
        K = tf.Variable(tf.truncated_normal([ksize, ksize, num_filters, in_shape[3]], stddev=0.1))
    else:
        # K = weights
        K = tf.transpose(weights, [0, 1, 3, 2])
    b = tf.Variable(tf.truncated_normal([num_filters]))
    in_shape = tf.shape(x)
    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
    l = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, stride, stride, 1], padding='SAME'), b)
    l = tf.add(l, tf.Variable(tf.zeros([K.get_shape().as_list()[2]])))
    l = tf.nn.relu(l, name=name)
    tf.add_to_collection('layers', l)
    return l


def shared_cnn(x):
    orig_shape = x.get_shape().as_list()
    summary_nodes = []

    with tf.variable_scope('encoder'):
        # W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.1))
        # b1 = tf.Variable(tf.truncated_normal([64]))
        # x = tf.add(tf.nn.conv2d(x, W1, strides=[1, 2, 2, 1], padding='SAME'), b1)

        # W2 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
        # b2 = tf.Variable(tf.truncated_normal([128]))
        # x = tf.add(tf.nn.conv2d(x, W2, strides=[1, 2, 2, 1], padding='SAME'), b2)

        # # 32x32x64
        # # 16x16x128
        
        x, W1 = _shared_cnn_downconv_layer(x, 64, 5, 2, 'Layer.Encoder.64')
        summary_nodes.append(tf.summary.histogram('Encoder 64', x))
        x, W2 = _shared_cnn_downconv_layer(x, 128, 5, 2, 'Layer.Encoder.128')
        summary_nodes.append(tf.summary.histogram('Encoder 128', x))
        x, W3 = _shared_cnn_downconv_layer(x, 256, 5, 2, 'Layer.Encoder.256')
        summary_nodes.append(tf.summary.histogram('Encoder 256', x))
        x, W4 = _shared_cnn_downconv_layer(x, 256, 5, 2, 'Layer.Encoder.256x2')
        summary_nodes.append(tf.summary.histogram('Encoder 256x2', x))
        x, W5 = _shared_cnn_downconv_layer(x, 96, 1, 1, 'Layer.Encoder.96')
        summary_nodes.append(tf.summary.histogram('Encoder 96', x))        
        x, W6 = _shared_cnn_downconv_layer(x, 32, 1, 1, 'Layer.Encoder.32')
        summary_nodes.append(tf.summary.histogram('Encoder 32', x))


    with tf.variable_scope('decoder'):
        x = _shared_cnn_downconv_layer(x, 96, 1, 1, 'Layer.Decoder.96') #, weights=W6)
        summary_nodes.append(tf.summary.histogram('Decoder 96', x))        
        x = _shared_cnn_downconv_layer(x, 256, 1, 1, 'Layer.Decoder.256') #, weights=W5)
        summary_nodes.append(tf.summary.histogram('Decoder 256', x))                
        x = _shared_cnn_upconv_layer(x, W4, 256, 5, 2, 'Layer.Decoder.256x2')
        summary_nodes.append(tf.summary.histogram('Decoder 256x2', x))                
        x = _shared_cnn_upconv_layer(x, W3, 128, 5, 2, 'Layer.Decoder.128')
        summary_nodes.append(tf.summary.histogram('Decoder 128', x))                
        x = _shared_cnn_upconv_layer(x, W2, 64, 5, 2, 'Layer.Decoder.64')
        summary_nodes.append(tf.summary.histogram('Decoder 64', x))                
        # output
        x = _shared_cnn_upconv_layer(x, W1, 3, 5, 2, 'Layer.Decoder.3')
        summary_nodes.append(tf.summary.histogram('Decoder 3', x))                        
        

    return x, tf.summary.merge(summary_nodes)
