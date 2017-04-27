import tensorflow as tf


def lrelu(x, leak=0.2, name='lrelu'):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def dense(x, input_size, output_size):
    W = tf.Variable(tf.random_normal([input_size, output_size]))
    b = tf.Variable(tf.random_normal([output_size]))
    return tf.matmul(x, W) + b
                        

def conv2d(x, input_size, output_size, ksize=3, stride=1, name=None):
    K = tf.Variable(tf.truncated_normal([ksize, ksize, input_size, output_size], stddev=0.1))
    h = tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME')
    b = tf.Variable(tf.truncated_normal([output_size]))    
    return h + b


def upsize(x, factor, output_size):
    input_shape = tf.shape(x)
    return tf.stack([input_shape[0], input_shape[1]*2, input_shape[2]*2, output_size])
    
    
def deconv2d(x, input_size, output_size, ksize=3, stride=2, name=None):
    K = tf.Variable(tf.truncated_normal([ksize, ksize, output_size, input_size], stddev=0.1))
    h = tf.nn.conv2d_transpose(x, K, output_shape=upsize(x, 2, output_size), strides=[1, stride, stride, 1], padding='SAME')
    b = tf.Variable(tf.truncated_normal([output_size]))
    return h + b


def flatten(x):
    input_shape = tf.shape(x)
    output_size = input_shape[1] * input_shape[2] * input_shape[3]
    return tf.reshape(x, [-1, output_size])



def L(x):
    """Mark this op as a layer."""
    tf.add_to_collection('layers', x)
    tf.summary.histogram(x.name, x)
    return x
    
