import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from math import ceil, sqrt
from models import simple_fc, simple_cnn
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
# parser.add_argument('--mode', type=str)
args = parser.parse_args()


sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))
graph = tf.get_default_graph()

# y_hat = graph.as_graph_element('outputs/decoder/Layer.Output').outputs[0]
# x_input = graph.as_graph_element('inputs/x_input').outputs[0]
# layer1 = graph.as_graph_element('outputs/encoder/Layer.Encoder.6').outputs[0]
# layer2 = graph.as_graph_element('outputs/decoder/Layer.Decoder.6').outputs[0]
#layer = graph.as_graph_element('outputs/encoder/Layer.Encoder.12').outputs[0]
#layer = graph.as_graph_element('outputs/encoder/Layer.Encoder.24').outputs[0]

# print(x_input, x_input.shape)
# print(y_hat, y_hat.shape)
# print(layer, layer.shape)


# visualize activations for a particular input
def visualize_activations(layer, input):
    x_input = graph.as_graph_element('inputs/x_input').outputs[0]
    activations = sess.run(layer, feed_dict={x_input: input})

    num_filters = activations.shape[-1]
    montage_w = ceil(sqrt(num_filters))
    montage_h = int(num_filters/montage_w)
    
    montage = []
    row_pos = 0
    for f_idx in range(num_filters):
        f = activations[0,:,:,f_idx] * 255.0
        f = np.expand_dims(f, 2)
        if row_pos % montage_w == 0:
            montage.append([f])
        else:
            montage[-1].append(f)
        row_pos += 1
        
    # create nxn montage image
    return np.vstack([np.hstack(row) for row in montage])

data = get_dataset('floorplan')
layer = graph.as_graph_element('outputs/encoder/Layer.Encoder.64').outputs[0]
result = visualize_activations(layer, np.expand_dims(data.test.images[0], 0))
print(type(result), result.shape)
cv2.imwrite('result.png', result)




# # visualize trained weights
# # for v in tf.trainable_variables():
# #     print(v)
# w1_name = 'outputs/encoder/Variable/read:0'
# w2_name = 'outputs/encoder/Variable_2/read:0'
# w3_name = 'outputs/encoder/Variable_4/read:0'
# w1 = graph.as_graph_element(w1_name)
# w_value = sess.run(w1)
# montage = None
# for i in range(w_value.shape[-1]):
#     f1 = w_value[:,:,:,i] * 255.0
#     print(f1.shape)
#     if montage is None:
#         montage = f1
#     else:
#         montage = np.hstack((montage, f1))
# cv2.imwrite('weights.png', montage)





# # # visualize image that most activates a filter via gradient ascent
# input_img = graph.as_graph_element('inputs/x_input').outputs[0]
# layer_output = graph.as_graph_element('outputs/encoder/Layer.Encoder.64').outputs[0]

# for idx in range(layer_output.get_shape()[-1]):
#     loss = tf.reduce_mean(layer_output[:,:,:,idx])
#     grads = tf.gradients(loss, input_img)[0]
#     input_img_data = np.random.random([1, 64, 64, 3])

#     for n in range(20):
#         loss_value, grads_value = sess.run([loss, grads], feed_dict={input_img: input_img_data})
#         print('grads:',grads_value.shape)
#         input_img_data += grads_value

#     input_img_data = np.squeeze(input_img_data)
#     cv2.imwrite('grad_ascent_{}.png'.format(idx), input_img_data * 255.0)





# # visualize image that most activates a filter
# xs = np.random.random([1, 64, 64, 3])
# # xs = np.random.random([1, 32, 32, 3])
# loss = tf.reduce_mean(layer1[:,:,:,0])
# # result = sess.run(layer1, feed_dict={x_input: xs})
# grads = tf.gradients(loss, [layer1])[0]

# val = sess.run(grads, feed_dict={x_input: xs})
# print(val.shape)



# print(f_val.shape)
# xs += f_val

# cv2.imwrite('grad_test.png', xs * 255.0)


# montage = None
# for f_idx in range(layer1.shape[-1]):
#     f = val[0, :, :, f_idx] * 255.0
#     f = np.expand_dims(f, 2)
#     if montage is None:
#         montage = f
#     else:
#         montage = np.hstack((montage, f))
# cv2.imwrite('grad_test.png', montage)



# c = sess.run([y_hat], {x_input: xs})

    
# cv2.imwrite('encoder_test.png', gen_montage(encoder_l))
# cv2.imwrite('decoder_test.png', gen_montage(decoder_l))




# print(c)
