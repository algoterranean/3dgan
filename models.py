import tensorflow as tf

def _fc_layer(x, x_size, y_size):
    W = tf.Variable(tf.random_normal([x_size, y_size]))
    b = tf.Variable(tf.random_normal([y_size]))
    return tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))


def simple_fc(x, layer_sizes):
    orig_shape = list(x.get_shape())

    with tf.variable_scope('input'):
        # flatten        
        x = tf.contrib.layers.flatten(x)
        flattened_size = int(list(x.get_shape())[1])
        print('input layer:', x, x.get_shape())
        
    with tf.variable_scope('encoder'):
        # encoder
        for size in layer_sizes:
            s = int(list(x.get_shape())[1])
            x = _fc_layer(x, s, size)
            print('encoder:', x, x.get_shape())

    with tf.variable_scope('decoder'):
        # decoder
        # layer_sizes = list(reversed(layer_sizes[1:]))
        # print('layer_sizes:', layer_sizes, layer_sizes[1::-1][1:])
        for size in layer_sizes[1::-1][1:]:
            s = int(list(x.get_shape())[1])
            x = _fc_layer(x, s, size)
            print('decoder:', x, x.get_shape())

        x = _fc_layer(x, int(list(x.get_shape())[1]), flattened_size)
        print('final layer:', x, x.get_shape())

    with tf.variable_scope('output'):
        # unflatten
        l = list(orig_shape)[1:]
        l = [-1, int(l[0]), int(l[1]), int(l[2])]
        print('reshape:', l)
        x = tf.reshape(x, l)

    return x

def _cnn_layer(x, x_size, y_size):
    # 64x64x1 > 64x64x64 > 32x32x128 > 16x16x256 < 32x32x128 < 64x64x64 < 64x64x1
    K = tf.Variable(tf.truncated_normal([3, 3, x_size, y_size], stddev=0.1))
    l = tf.nn.conv2d(x, K, strides=[1, 1, 1, 1], padding='SAME')
    # l = tf.nn.max_pool(l, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return l

def simple_cnn(x, layer_sizes):
    orig_shape = list(x.get_shape())

    print('i)', x.get_shape())

    # encoder
    with tf.variable_scope('encoder'):
        for size in layer_sizes:
            s = int(list(x.get_shape())[3])
            K = tf.Variable(tf.truncated_normal([3, 3, s, size], stddev=0.1))
            x = tf.nn.conv2d(x, K, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.relu(x)
            print('e)', x.get_shape())
            # x = _cnn_layer(x, s, size)

    # decoder
    with tf.variable_scope('decoder'):
        layer_sizes = reversed(layer_sizes)
        for size in layer_sizes:
            s = int(list(x.get_shape())[3])
            K = tf.Variable(tf.truncated_normal([3, 3, s, size], stddev=0.1))
            os = [-1] + list(x.get_shape())[1:]
            os = [os[0], int(os[1])*2, int(os[2])*2, size]
            # print('GET SHAPE:', os)
            x = tf.nn.conv2d_transpose(x, K, output_shape=os, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.relu(x)
            # x = _cnn_layer(x, s, size)
            print('d)', x.get_shape())

        s = int(list(x.get_shape())[3])
        size = int(orig_shape[3])
        K = tf.Variable(tf.truncated_normal([3, 3, s, size], stddev=0.1))

        os = [-1] + list(x.get_shape())[1:]
        os = [os[0], int(os[1])*2, int(os[2])*2, int(os[3])]
        # print('GET SHAPE:', os)
        x = tf.nn.conv2d_transpose(x, K, output_shape=os, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x)
        print('o)', x.get_shape())
        # x = tf.nn.conv2d(x, K, strides=[1, 1, 1, 1], padding='SAME')
            
            
        # x = _cnn_layer(x, i, i)
        
    return x
