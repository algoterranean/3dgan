import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from models import simple_fc, simple_cnn
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()

sess = tf.Session()


saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))


graph = tf.get_default_graph()

y_hat = graph.as_graph_element('outputs/decoder/Layer.Output').outputs[0]
x_input = graph.as_graph_element('inputs/x_input').outputs[0]
layer = graph.as_graph_element('outputs/encoder/Layer.Encoder.6').outputs[0]
#layer = graph.as_graph_element('outputs/encoder/Layer.Encoder.12').outputs[0]
#layer = graph.as_graph_element('outputs/encoder/Layer.Encoder.24').outputs[0]

# xs = np.random.random([1, 64, 64, 3])
# c = sess.run([y_hat], {x_input: xs})
# print(c)

print(x_input, x_input.shape)
print(y_hat, y_hat.shape)
print(layer, layer.shape)

data = get_dataset('floorplan')

xs = data.test.images[0]
xs = np.expand_dims(xs, 0)



# visualize activations for a particular input
l = sess.run(layer, feed_dict={x_input: xs})
print(layer.shape)
print(l.shape)

montage = None
for f_idx in range(l.shape[-1]):
    f = l[0,:,:,f_idx] * 255.0
    f = np.expand_dims(f, 2)
    if montage is None:
        montage = f
    else:
        montage = np.hstack((montage, f))
cv2.imwrite('test.png', montage)
