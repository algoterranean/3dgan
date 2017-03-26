import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
import datetime
from math import ceil, sqrt
from itertools import chain
# 
from models import simple_fc, simple_cnn
from util import *



parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()

sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))
graph = tf.get_default_graph()


# usage: list(chunks(some_list, chunk_size)) ==> list of lists of that size
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



# TODO: Add option to use color for a border
def stitch_montage(image_list, add_border=False):
    """Stitch a list of equally-shaped images into a single image."""
    num_images = len(image_list)
    montage_w = ceil(sqrt(num_images))
    montage_h = int(num_images/montage_w)
    ishape = image_list[0].shape
    # black borders
    v_border = np.zeros((ishape[0], 1, ishape[-1]))
    h_border = np.zeros((1, (ishape[1]+1) * montage_w + 1, ishape[-1]))

    montage = list(chunks(image_list, montage_w))
    # fill in any remaining missing images in square montage
    remaining = montage_w - (num_images - montage_w * montage_h)
    if remaining < montage_w:
        # dummy_shape = weights[:,:,:,0].shape if rgb else np.expand_dims(weights[:,:,0,0], 2).shape
        for _ in range(remaining):
            montage[-1].append(np.zeros(ishape))

    if add_border:
        b = [v_border for x in range(len(montage[0]))]
        c = [h_border for x in range(len(montage))]        
        return np.concatenate(list(chain(*zip(c, [np.concatenate(list(chain(*zip(b, row))) + [v_border], axis=1) for row in montage]))) + [h_border], axis=0)
    else:
        return np.concatenate([np.concatenate(row, axis=1) for row in montage], axis=0)
    

# TODO: Add option for x_input, or find it dynamically
def visualize_activations(layer, input):
    """Generate image of layer's activations given a specific input."""
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
# layers = tf.get_collection('layers')
# results = visualize_all_activations(layers, data.test.images[0])
# for i in range(len(results)):
#     cv2.imwrite('result_layer_' + str(i) + '.png', results[i])



# TODO: Given a layer, find the weights
# visualize trained weights
def visualize_weights(var):
    """Generate image of the weights of a layer."""
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
# # note, the only variables we are interested in are the conv weights, which have shape of length 4
# weight_vars = [v for v in tf.trainable_variables() if len(v.get_shape()) == 4]
# results = visualize_all_weights(weight_vars)
# for i in range(len(results)):
#     print('results', i, results[i].shape)
#     cv2.imwrite('weights_' + str(i) + '.png', results[i])


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# visualize image that most activates a filter via gradient ascent
def visualize_bestfit_image(layer):
    """Use gradient ascent to find image that best activates a layer's filters."""
    x_input = graph.as_graph_element('inputs/x_input').outputs[0]
    dt = datetime.datetime.now()

    image_list = []
    for idx in range(layer.get_shape()[-1]):
        with tf.device("/gpu:0"):
            dt_f = datetime.datetime.now()
            # start with noise averaged around gray
            input_img_data = np.random.random([1, 64, 64, 3])
            input_img_data = (input_img_data - 0.5) * 20 + 128.0
            
            loss = tf.reduce_mean(layer[:,:,:,idx])
            grads = tf.gradients(loss, x_input)[0]
            # normalize gradients
            grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)


            for n in range(20):
                loss_value, grads_value = sess.run([loss, grads], feed_dict={x_input: input_img_data})
                input_img_data += grads_value
                
            image_list.append(deprocess_image(input_img_data[0]))
            print('Completed filter {}/{}, {} elapsed'.format(idx, layer.get_shape()[-1], datetime.datetime.now() - dt_f))

    print('Finished', layer)
    print('Elapsed: {}'.format(datetime.datetime.now() - dt))
    return stitch_montage(image_list, add_border=True)
    

def visualize_all_bestfit_images(layers):
    return [visualize_bestfit_image(layer) for layer in layers]


# # how to use:
# i = 0
# for layer in tf.get_collection('layers'):
#     cv2.imwrite('test_{}.png'.format(i), visualize_bestfit_image(layer))
#     i += 1

layer = tf.get_collection('layers')[0]
result = visualize_bestfit_image(layer)
cv2.imwrite('test.png', result)

# layers = tf.get_collection('layers')
# results = visualize_all_bestfit_images(layers)
# for i in range(len(results)):
#     cv2.imwrite('bestfit_' + str(i) + '.png', results[i])


    
# encoder 256: 18.5 min
# encoder 256x2: 37.43 min
# encoder 96: 20.5 min
# encoder 32: 8 min
# decoder 96: 27min, 11 sec
# decoder 256: 1hr 41min

