import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav_init
from tensorflow.contrib.layers import variance_scaling_initializer as he_init
from tensorflow.contrib.layers import batch_norm


def lrelu(x, leak=0.2, name='lrelu'):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    out = f1 * x + f2 * abs(x)
    return tf.identity(out, name=name)


def dense(x, input_size, output_size, reuse=False, name=None):
    w_name = name if name is None else name + '_w'
    b_name = name if name is None else name + '_b'

    with tf.variable_scope('vars', reuse=reuse):            
        W = tf.get_variable(name=w_name, shape=[input_size, output_size], initializer=he_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=he_init())
    return tf.matmul(x, W) + b
                        

def conv2d(x, input_size, output_size, ksize=3, stride=1, reuse=False, name=None):
    w_name = name if name is None else name + '_w'
    b_name = name if name is None else name + '_b'
    
    with tf.variable_scope('vars', reuse=reuse):
        K = tf.get_variable(name=w_name, shape=[ksize, ksize, input_size, output_size], initializer=he_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=he_init())
    h = tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME')
    return h + b


def upsize(x, factor, output_size):
    input_shape = tf.shape(x)
    return tf.stack([input_shape[0], input_shape[1]*factor, input_shape[2]*factor, output_size])
    
    
def deconv2d(x, input_size, output_size, ksize=3, stride=2, reuse=False, name=None):
    w_name = name if name is None else name + '_w'
    b_name = name if name is None else name + '_b'
    
    with tf.variable_scope('vars', reuse=reuse):
        K = tf.get_variable(name=w_name, shape=[ksize, ksize, output_size, input_size], initializer=he_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=he_init())
    h = tf.nn.conv2d_transpose(x, K, output_shape=upsize(x, 2, output_size), strides=[1, stride, stride, 1], padding='SAME')
    return h + b


def flatten(x, name=None):
    input_shape = tf.shape(x)
    output_size = input_shape[1] * input_shape[2] * input_shape[3]
    return tf.reshape(x, [-1, output_size], name=name)


def montage_summary(x, args, name='montage'):
    # batch_size = tf.shape(x[0])[0]
    with tf.variable_scope('montage_summary'):
        images = tf.unstack(x, num=args.batch_size, axis=0)[0:args.examples]

        image = tf.concat(images, axis=1)
        image = tf.expand_dims(image, axis=0)
        return tf.summary.image('montage', image)


def input_slice(x, batch_size, gpu_id):
    with tf.variable_scope('input_slice'):
        slice = x[gpu_id * batch_size : (gpu_id+1)*batch_size, :]
        # slice = tf.slice(x, gpu_id * batch_size, batch_size)
    return slice

    
