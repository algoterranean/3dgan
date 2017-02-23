import tensorflow as tf


def _layer(x, x_size, y_size):
    W = tf.Variable(tf.random_normal([x_size, y_size]))
    b = tf.Variable(tf.random_normal([y_size]))
    return tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))


def model(x, layer_sizes):
    orig_shape = list(x.get_shape())
    
    # encoder
    for size in layer_sizes:
        s = int(list(x.get_shape())[1])
        x = _layer(x, s, size)

    # decoder
    layer_sizes = reversed(layer_sizes[1:])
    for size in layer_sizes:
        s = int(list(x.get_shape())[1])
        x = _layer(x, s, size)
    x = _layer(x, int(list(x.get_shape())[1]), int(orig_shape[1]))

    return x
        
    # for idx, size in enumerate(layer_sizes):
    #     if idx < len(layer_sizes) - 1:
    #         print(size, layer_sizes[idx+1], x.get_shape())
    #         x = _layer(x, size, layer_sizes[idx+1])

    # # decoder
    # for idx, size in enumerate(layer_sizes[::-1]):
    #     if idx > 0:
    #         x = _layer(x, size, layer_sizes[idx-1])
    #         print(size, layer_sizes[idx-1])
    # return x
                


