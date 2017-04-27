import tensorflow as tf
# from functools import
from .model import Model



class ChenCNN(Model):
    def __init__(self, x):
        Model.__init__(self, x)

    def _build_graph(self, x):
        orig_shape = x.get_shape().as_list()
        summary_nodes = []

        # encoder
        with tf.variable_scope('encoder'):
            # layer_sizes = [64, 128, 256, 256, 96, 32]
            x = self._down(x, 64, 5, 2, 'Layer.Encoder.64')
            x = self._down(x, 128, 5, 2, 'Layer.Encoder.128')
            x = self._down(x, 256, 5, 2, 'Layer.Encoder.256')
            x = self._down(x, 256, 5, 2, 'Layer.Encoder.256x2')
            x = self._down(x, 96, 1, 1, 'Layer.Encoder.96')
            x = self._down(x, 32, 1, 1, 'Layer.Encoder.32')

        # decoder
        with tf.variable_scope('decoder'):
            x = self._down(x, 96, 1, 1, 'Layer.Decoder.96')
            x = self._down(x, 256, 1, 1, 'Layer.Decoder.256')
            x = self._up(x, 256, 5, 2, 'Layer.Decoder.256x2')
            x = self._up(x, 128, 5, 2, 'Layer.Decoder.128')
            x = self._up(x, 64, 5, 2, 'Layer.Decoder.64')
            # output
            x = self._up(x, 3, 5, 2, 'Layer.Decoder.3')

        self.output = x
        # return x, tf.summary.merge(summary_nodes)


    def _down(self, x, num_filters, ksize, stride, name):
        in_shape = x.get_shape().as_list()
        K = tf.Variable(tf.truncated_normal([ksize, ksize, in_shape[3], num_filters], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([num_filters]))
        l = tf.add(tf.nn.conv2d(x, K, strides=[1, stride, stride, 1], padding='SAME'), b)
        
        l = tf.contrib.layers.batch_norm(l)
        l = tf.nn.relu(l, name=name)
        tf.add_to_collection('layers', l)
        self._add_summary(l, name)
        return l

    def _up(self, x, num_filters, ksize, stride, name):
        in_shape = x.get_shape().as_list()
        K = tf.Variable(tf.truncated_normal([ksize, ksize, num_filters, in_shape[3]], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([num_filters]))
        in_shape = tf.shape(x)
        out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, num_filters])
        l = tf.add(tf.nn.conv2d_transpose(x, K, output_shape=out_shape, strides=[1, stride, stride, 1], padding='SAME'), b)
        l = tf.add(l, tf.Variable(tf.zeros([K.get_shape().as_list()[2]])))
        l = tf.nn.relu(l, name=name)
        tf.add_to_collection('layers', l)
        self._add_summary(l, name)
        return l    

