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


def get_layer(name):
    op = graph.as_graph_element(name)
    if op is not None:
        return op.outputs[0]
    else:
        return None


# usage: list(chunks(some_list, chunk_size)) ==> list of lists of that size
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

        
def stitch_montage(image_list):
    num_images = len(image_list)
    montage_w = ceil(sqrt(num_images))
    montage_h = int(num_images/montage_w)
    
    montage = list(chunks(image_list, montage_w))
    # fill in any remaining missing images in square montage
    remaining = montage_w - (num_images - montage_w * montage_h)
    if remaining < montage_w:
        dummy_shape = image_list[0].shape
        # dummy_shape = weights[:,:,:,0].shape if rgb else np.expand_dims(weights[:,:,0,0], 2).shape
        for _ in range(remaining):
            montage[-1].append(np.zeros(dummy_shape))
    return np.vstack([np.hstack(row) for row in montage])
    


# visualize activations for a particular input
def visualize_activations(layer, input):
    # input is a single image, so put it into batch form for sess.run
    input = np.expand_dims(input, 0)
    x_input = graph.as_graph_element('inputs/x_input').outputs[0]
    activations = sess.run(layer, feed_dict={x_input: input})

    image_list = []
    for f_idx in range(activations.shape[-1]):
        f = activations[0,:,:,f_idx] * 255.0
        image_list.append(np.expand_dims(f, 2))
    return stitch_montage(image_list)


def visualize_all_activations(layers, input):
    return [visualize_activations(layer, input) for layer in layers]

# # how to use:
# data = get_dataset('floorplan')
# layers = tf.get_collection(key='layers')
# results = visualize_all_activations(layers, data.test.images[0])
# for i in range(len(results)):
#     cv2.imwrite('result_layer_' + str(i) + '.png', results[i])





# visualize trained weights
def visualize_weights(var):
    weights = sess.run(var)
    rgb = weights.shape[-2] == 3 # should output be rgb or grayscale?
    num_filters = weights.shape[-1] if rgb else weights.shape[-1] * weights.shape[-2]

    image_list = []
    for f_idx in range(weights.shape[-1]):
        if rgb:
            image_list.append(weights[:,:,:,f_idx]*255.0)
        else:
            for f_idx2 in range(weights.shape[-2]):
                f = weights[:,:,f_idx2,f_idx] * 255.0
                image_list.append(np.expand_dims(f, 2))
    return stitch_montage(image_list)


def visualize_all_weights(weights):
    return [visualize_weights(var) for var in weights]

# # how to use:
# weight_vars = [v for v in tf.trainable_variables() if len(v.get_shape()) == 4]
# results = visualize_all_weights(weight_vars)
# for i in range(len(results)):
#     print('results', i, results[i].shape)
#     cv2.imwrite('weights_' + str(i) + '.png', results[i])




# visualize image that most activates a filter via gradient ascent


# # # how to use:
# # layers = tf.get_collection(key='layers')
# input_img = graph.as_graph_element('inputs/x_input').outputs[0]
# layer_output = graph.as_graph_element('outputs/encoder/Layer.Encoder.64').outputs[0]
# visualize_bestfit_image(layer_output)


# def visualize_bestfit_image(layer):
#     for idx in range(layer.get_shape()[-1]):
#         input_img_data = np.random.random([1, 64, 64, 3])
#         loss = tf.reduce_mean(layer[:,:,:,idx])
#         grads = tf.gradients(loss, input_img)[0]

#         for n in range(20):
#             loss_value, grads_value = sess.run([loss, grads], feed_dict={input_img: input_img_data})
#             print('grads:',grads_value.shape)
#             input_img_data += grads_value

#         input_img_data = np.squeeze(input_img_data)
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
