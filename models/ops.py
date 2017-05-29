import tensorflow as tf
import math
import re
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

        #if input_size == 3 or input_size == 1:
            #K_t = tf.transpose(K, [3, 0, 1, 2])
            #images = tf.unstack(K_t, num=output_size, axis=0)
            # print('num images', output_size)
            # num_blanks = int(math.sqrt(output_size))**2 - output_size
            # print('num_blanks', num_blanks)
            # new_images = images + [np.zeros(images[0].shape) for n in range(num_blanks)]
            # print('new image shape', images[0].shape)

            # # chunks(new_images, int(math.sqrt(output_size)))

            #m = tf.split(images, int(math.sqrt(output_size)), axis=1)
            #tf.concat(m, axis=0)

            # r = tf.expand_dims(tf.concat(images, axis=1), axis=0)
            
            #tf.summary.image(w_name, m)
        # images = tf.unstack(K_t, num=output_size, axis=0)
        # image = tf.concat(images, axis=1)
        # image = tf.expand_dim


        # print('weights:', K.shape)
        # a = 
        # print('a', a.shape)        
        # print('biases:', b.shape)
        
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



def activation_summary(x, rows=0, cols=0, montage=True):
    n = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.summary.histogram(n + '/activations', x)
    tf.summary.scalar(n + '/sparsity', tf.nn.zero_fraction(x))
    if montage:
        montage_summary(tf.transpose(x[0], [2, 0, 1]), rows, cols, n + '/activtions')
    

def montage_summary(x, num_cols, num_rows, name='montage'):
    num_examples = num_cols * num_rows
    with tf.variable_scope(name):
        images = x[0:num_examples] #, :, :, :]
        images = tf.split(images, num_cols, axis=0)
        images = tf.concat(images, axis=1)
        images = tf.unstack(images, num_rows, axis=0)
        images = tf.concat(images, axis=1)

        
        if len(images.shape) < 3:
            images = tf.expand_dims(images, axis=2)
        # else:
        #     print('what shape for', name, images.shape)
            
            
        images = tf.expand_dims(images, axis=0)
        return tf.summary.image('montage', images)



def input_slice(x, batch_size, gpu_id):
    with tf.variable_scope('input_slice'):
        slice = x[gpu_id * batch_size : (gpu_id+1)*batch_size, :]
        # slice = tf.slice(x, gpu_id * batch_size, batch_size)
    return slice

    
