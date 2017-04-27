import tensorflow as tf




def downconv_layer(x, num_filters, k_size=3, activation=tf.nn.relu, max_pool=0, stride=1, name=None):
    with tf.name_scope("layers") as scope:
        in_shape = x.get_shape().as_list()
        # weights and bias        
        K = tf.Variable(tf.truncated_normal([k_size, k_size, in_shape[3], num_filters], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([num_filters]))
        # conv
        conv = tf.add(tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME'), b)
        # activation and pooling
        conv = activation(conv)
        if max_pool > 0:
            conv = tf.nn.max_pool(conv, [1, max_pool, max_pool, 1], strides=[1, max_pool, max_pool, 1], padding='SAME')
            
        return tf.identity(conv, name=scope)

    
def upconv_layer(x, num_filters, k_size=3, activation=tf.nn.relu, name=None):
    with tf.name_scope("layers") as scope:
        in_shape = x.get_shape().as_list()
        # weights and bias
        K = tf.Variable(tf.truncated_normal([k_size, k_size, num_filters, in_shape[3]], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([num_filters]))
        # calculate dynamic shape (needed for for upconv op)
        in_shape = tf.shape(x)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
        # conv and activation
        conv = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME'), b)
        conv = tf.add(conv, tf.Variable(tf.zeros([K.get_shape().as_list()[2]]))) # ?
        conv = activation(conv)
        
        return tf.identity(conv, name=scope)

        
    





# def downconv_layer(x, num_filters, name):
#     in_shape = x.get_shape().as_list()
#     # weights and bias
#     K = tf.Variable(tf.truncated_normal([3, 3, in_shape[3], num_filters], stddev=0.1))
#     b = tf.Variable(tf.truncated_normal([num_filters]))
#     # layer
#     l = tf.add(tf.nn.conv2d(x, K, strides=[1, 1, 1, 1], padding='SAME'), b)
#     l2 = tf.nn.relu(l)
#     l = tf.nn.max_pool(l2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
#     return l


# def upconv_layer(x, num_filters, name):
#     in_shape = x.get_shape().as_list()
#     # weights and bias
#     K = tf.Variable(tf.truncated_normal([3, 3, num_filters, in_shape[3]], stddev=0.1))
#     b = tf.Variable(tf.truncated_normal([num_filters]))
    
#     # layer
#     in_shape = tf.shape(x)
#     out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
    
#     l = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME'), b)
#     l = tf.add(l, tf.Variable(tf.zeros([K.get_shape().as_list()[2]])))
#     l = tf.nn.relu(l, name=name)
#     return l




# def _chen_downconv_layer(x, num_filters, ksize, stride, name):
#     in_shape = x.get_shape().as_list()
#     K = tf.Variable(tf.truncated_normal([ksize, ksize, in_shape[3], num_filters], stddev=0.1))
#     b = tf.Variable(tf.truncated_normal([num_filters]))
#     l = tf.add(tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME'), b)
#     l = tf.contrib.layers.batch_norm(l)
#     l = tf.nn.relu(l, name=name)
#     tf.add_to_collection('layers', l)
#     return l

# def _chen_upconv_layer(x, num_filters, ksize, stride, name):
#     in_shape = x.get_shape().as_list()
#     K = tf.Variable(tf.truncated_normal([ksize, ksize, num_filters, in_shape[3]], stddev=0.1))
#     b = tf.Variable(tf.truncated_normal([num_filters]))
#     in_shape = tf.shape(x)
#     out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
#     l = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, stride, stride, 1], padding='SAME'), b)
#     l = tf.add(l, tf.Variable(tf.zeros([K.get_shape().as_list()[2]])))
#     l = tf.nn.relu(l, name=name)
#     tf.add_to_collection('layers', l)
#     return l
