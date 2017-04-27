import tensorflow as tf
from .model import Model

class SimpleFC(Model):
    def __init__(self, x, layer_sizes):
        self.layer_sizes = layer_sizes
        Model.__init__(x)

    def _fc_layer(x, x_size, y_size):
        W = tf.Variable(tf.random_normal([x_size, y_size]))
        b = tf.Variable(tf.random_normal([y_size]))
        l = tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))
        return l
        

    def _build_graph(self, x):
        orig_shape = list(x.get_shape())
        with tf.variable_scope('input'):
            # flatten        
            x = tf.contrib.layers.flatten(x)
            flattened_size = int(list(x.get_shape())[1])
            self.summary_nodes.append(tf.summary.histogram('Input', x))
        
        with tf.variable_scope('encoder'):
            # encoder
            for size in layer_sizes:
                s = int(list(x.get_shape())[1])
                x = _fc_layer(x, s, size)
                self.summary_nodes.append(tf.summary.histogram('Encoder {}'.format(size), x))

        with tf.variable_scope('decoder'):
            # decoder
            for size in layer_sizes[1::-1][1:]:
                s = int(list(x.get_shape())[1])
                x = _fc_layer(x, s, size)
                self.summary_nodes.append(tf.summary.histogram('Decoder {}'.format(size), x))
            x = _fc_layer(x, int(list(x.get_shape())[1]), flattened_size)
            self.summary_nodes.append(tf.summary.histogram('Output', x))

        with tf.variable_scope('output'):
            # unflatten
            l = list(orig_shape)[1:]
            l = [-1, int(l[0]), int(l[1]), int(l[2])]
            x = tf.reshape(x, l)

        self.model = x
        
        



