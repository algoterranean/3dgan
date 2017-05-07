import tensorflow as tf
from .model import Model
from .ops import L, flatten, dense, lrelu

class SimpleFC(Model):
    def __init__(self, optimizer, x, layer_sizes):
        Model.__init__(self)
        self.layer_sizes = layer_sizes
        self.orig_shape = x.get_shape().as_list()
        self.orig_size = self.orig_shape[1] * self.orig_shape[2] * self.orig_shape[3]
        
        
        self._encoder = self._build_encoder(x)
        self._decoder = self._build_decoder(self._encoder)
        self._optimizer = optimizer
        self._loss = tf.reduce_mean(tf.abs(x - self._decoder))
        self._train_op = optimizer.minimize(self._loss)
        


    def _build_encoder(self, x):
        with tf.variable_scope('encoder'):
            x = flatten(x)
            in_size = self.orig_size
            for out_size in self.layer_sizes:
                x = lrelu(dense(x, in_size, out_size));     L(x)
                in_size = out_size
            tf.identity(x, name='sample')
        return x
    

    def _build_decoder(self, x):
        in_size = x.get_shape().as_list()[1]
        
        with tf.variable_scope('decoder'):
            for out_size in self.layer_sizes[1::-1]:
                x = lrelu(dense(x, in_size, out_size));     L(x)
                in_size = out_size

            # resize to correct output size
            # TODO why?
            x = lrelu(dense(x, in_size, self.orig_size));   L(x)
            x = tf.reshape(x, [-1, 64, 64, 3]);
            #self.orig_shape) #[-1, self.orig_shape[0], self.orig_shape[1], self.orig_shape[2]])
            tf.identity(x, name='sample')
        return x
                

            
        #     for size in self.layer_sizes[1::-1][1:]:
        #         s = int(list(x.get_shape())[1])
        #         x = _fc_layer(x, s, size);       L(x)

        #     flattened_size = orig_shape[1] * orig_shape[2] * orig_shape[3]
        #     x = _fc_layer(x, int(list(x.get_shape())[1]), flattened_size)

        #     # unflatten
        #     l = list(orig_shape)[1:]
        #     l = [-1, int(l[0]), int(l[1]), int(l[2])]
        #     x = tf.reshape(x, l)
        #     tf.identity(x, name='sample')
        # return x


    # def _fc_layer(self, x, x_size, y_size):
    #     W = tf.Variable(tf.random_normal([x_size, y_size]))
    #     b = tf.Variable(tf.random_normal([y_size]))
    #     l = tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))
    #     return l
        

    # def _build_graph(self, x):
    #     orig_shape = list(x.get_shape())
    #     with tf.variable_scope('input'):
    #         # flatten        
    #         x = tf.contrib.layers.flatten(x)
    #         flattened_size = int(list(x.get_shape())[1])
    #         self.summary_nodes.append(tf.summary.histogram('Input', x))
        
    #     with tf.variable_scope('encoder'):
    #         # encoder
    #         for size in layer_sizes:
    #             s = int(list(x.get_shape())[1])
    #             x = _fc_layer(x, s, size)
    #             self.summary_nodes.append(tf.summary.histogram('Encoder {}'.format(size), x))

    #     with tf.variable_scope('decoder'):
    #         # decoder
    #         for size in layer_sizes[1::-1][1:]:
    #             s = int(list(x.get_shape())[1])
    #             x = _fc_layer(x, s, size)
    #             self.summary_nodes.append(tf.summary.histogram('Decoder {}'.format(size), x))
    #         x = _fc_layer(x, int(list(x.get_shape())[1]), flattened_size)
    #         self.summary_nodes.append(tf.summary.histogram('Output', x))

    #     with tf.variable_scope('output'):
    #         # unflatten
    #         l = list(orig_shape)[1:]
    #         l = [-1, int(l[0]), int(l[1]), int(l[2])]
    #         x = tf.reshape(x, l)

    #     self.model = x
        
        



