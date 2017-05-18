import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as x_init
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

    # input_size = inttf.shape(x)[1])
    # var_shape = tf.stack([input_size, output_size])

    # with tf.device('/cpu:0'):
    with tf.variable_scope('vars', reuse=reuse):            
        W = tf.get_variable(name=w_name, shape=[input_size, output_size], initializer=he_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=he_init())
    return tf.matmul(x, W) + b
                        

def conv2d(x, input_size, output_size, ksize=3, stride=1, reuse=False, name=None):
    w_name = name if name is None else name + '_w'
    b_name = name if name is None else name + '_b'
    
    # if not name is None:
    # with tf.device('/cpu:0'):
    with tf.variable_scope('vars', reuse=reuse):
        K = tf.get_variable(name=w_name, shape=[ksize, ksize, input_size, output_size], initializer=he_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=he_init())
    # else:
    #     K = tf.Variable(tf.truncated_normal([ksize, ksize, input_size, output_size], stddev=0.1)) #, name=name))
    #     b = tf.Variable(tf.truncated_normal([output_size]))
    h = tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME')
    # return batch_norm(h+b)
    # print('created var', name)
    return h + b


def upsize(x, factor, output_size):
    input_shape = tf.shape(x)
    return tf.stack([input_shape[0], input_shape[1]*2, input_shape[2]*2, output_size])
    
    
def deconv2d(x, input_size, output_size, ksize=3, stride=2, reuse=False, name=None):
    w_name = name if name is None else name + '_w'
    b_name = name if name is None else name + '_b'
    # with tf.device('/cpu:0'):
    with tf.variable_scope('vars', reuse=reuse):
        K = tf.get_variable(name=w_name, shape=[ksize, ksize, output_size, input_size], initializer=he_init())
        b = tf.get_variable(name=b_name, shape=[output_size], initializer=he_init())
    
    # K = tf.Variable(tf.truncated_normal([ksize, ksize, output_size, input_size], stddev=0.1))
    h = tf.nn.conv2d_transpose(x, K, output_shape=upsize(x, 2, output_size), strides=[1, stride, stride, 1], padding='SAME')
    # b = tf.Variable(tf.truncated_normal([output_size]))
    return h + b


def flatten(x, name=None):
    input_shape = tf.shape(x)
    output_size = input_shape[1] * input_shape[2] * input_shape[3]
    return tf.reshape(x, [-1, output_size], name=name)



def L(x):
    """Mark this op as a layer."""
    tf.add_to_collection('layers', x)
    tf.summary.histogram(x.name, x)
    return x


def M(x, collection):
    """Mark this op as part of a collection and add histograms."""
    tf.add_to_collection(collection, x)
    # add summary nodes
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))
    # tf.summary.image(x.op.name + '/activations', x, collections=['epoch'])
    return x


# def activation_summary(x):
#     tf.summary.histogram(x.op.name + '/activations', x, collections=['epoch'])
#     tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x), collections=['epoch'])
    
    
# def activation_summary(x):
#     tf.summary.image(x.op.name + '/activations', x, collections=['epoch'])
#     tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x), collections=['epoch'])
