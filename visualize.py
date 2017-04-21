# stdlib/external
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import datetime
from math import ceil, sqrt
from itertools import chain
# local
# from models.fc import simple_fc
# from models.conv import simple_cnn
# from models.chen import chen_cnn
# from models.shared_cnn import shared_cnn
from util import *



################################################
# utility/helper functions

def reload_session(dir, fn=None):
    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(dir, 'model'))
    if fn is None:
        chk_file = tf.train.latest_checkpoint(os.path.join(dir, 'checkpoints'))
    else:
        chk_file = fn
    # print('latest checkpoint:', latest)
    saver.restore(sess, chk_file)
    return sess


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
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# usage: list(chunks(some_list, chunk_size)) ==> list of lists of that size
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def stitch_montage(image_list, add_border=True, use_width=0):
    """Stitch a list of equally-shaped images into a single image."""
    num_images = len(image_list)
    if use_width > 0:
        montage_w = use_width
    else:
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

    

################################################
# visualization functions


def visualize_loss(dir):
    log_dir = os.path.join(dir, 'logs')
    train_loss_csv = np.genfromtxt(os.path.join(log_dir, 'train_loss.csv'), delimiter=',')
    test_loss_csv = np.genfromtxt(os.path.join(log_dir, 'test_loss.csv'), delimiter=',')
    validate_loss_csv = np.genfromtxt(os.path.join(log_dir, 'validate_loss.csv'), delimiter=',')
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family':'serif','serif':['Palatino']})
    for x in [(train_loss_csv, {}), (validate_loss_csv, {'color': 'firebrick'})]:
        data, plot_args = x
        iters = data[:,[0]]
        vals = data[:,[1]]
        plt.plot(iters, vals, **plot_args)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\ell_1$ Loss')
    plt.savefig(os.path.join(dir, 'images', 'loss.pdf'))
    plt.close()
    
    

def visualize_activations(layer, input):
    """Generate image of layer's activations given a specific input."""
    # input is a single image, so put it into batch form for sess.run
    input = np.expand_dims(input, 0)
    graph = tf.get_default_graph()
    x_input = graph.as_graph_element('inputs/x_input').outputs[0]
    activations = sess.run(layer, feed_dict={x_input: input})
    print('layer:', layer)
    print('activations:', activations.shape)

    image_list = []
    for f_idx in range(activations.shape[-1]):
        f = activations[0,:,:,f_idx] * 255.0
        image_list.append(np.expand_dims(f, 2))
    return stitch_montage(image_list)


def visualize_all_activations(layers, input):
    return [visualize_activations(layer, input) for layer in layers]


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


def visualize_timelapse(workspace_dir, example_images):
    # get list of checkpoint files in order
    checkpoint_files = []
    for f in os.listdir(os.path.join(workspace_dir, 'checkpoints')):
        if f.endswith('.meta'):
            checkpoint_files.append(f[:f.rfind('.')])
    checkpoint_files.sort()

    montage = [x*255.0 for x in example_images]
    for f in checkpoint_files:
        sess = reload_session(workspace_dir, os.path.join(workspace_dir, 'checkpoints', f))
        graph = tf.get_default_graph()
        x_input = graph.as_graph_element('inputs/x_input').outputs[0]
        y_hat = graph.as_graph_element('outputs/y_hat').outputs[0]
        
        results = sess.run(y_hat, feed_dict={x_input: example_images})
        for r in results:
            montage.append(r * 255.0)
    return stitch_montage(montage, use_width=len(example_images)) #args.examples)


# visualize image that most activates a filter via gradient ascent
def visualize_bestfit_image(layer):
    """Use gradient ascent to find image that best activates a layer's filters."""
    graph = tf.get_default_graph()
    x_input = graph.as_graph_element('inputs/x_input').outputs[0]
    dt = datetime.datetime.now()

    image_list = []
    for idx in range(layer.get_shape()[-1]):
        with tf.device("/gpu:0"):
            dt_f = datetime.datetime.now()
            # start with noise averaged around gray
            input_img_data = np.random.random([1, 64, 64, 3])
            input_img_data = (input_img_data - 0.5) * 20 + 128.0
                
            # build loss and gradients for this filter
            loss = tf.reduce_mean(layer[:,:,:,idx])
            grads = tf.gradients(loss, x_input)[0]
            # normalize gradients
            grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)


            # perform gradient ascent in image space
            for n in range(20):
                loss_value, grads_value = sess.run([loss, grads], feed_dict={x_input: input_img_data})
                input_img_data += grads_value
                # apply regularizer
                # gaussian
                if n % 4 == 0:
                    input_img_data = np.squeeze(input_img_data)
                    input_img_data = cv2.GaussianBlur(input_img_data, (3, 3), 0.5)
                    input_img_data = np.expand_dims(input_img_data, 0)
                # l2 decay
                input_img_data *= (1 - 0.0001)
                
                if loss_value <= 0:
                    input_img_data = np.ones([1, 64, 64, 3])
                    break
            image_list.append(deprocess_image(input_img_data[0]))

            

    return stitch_montage(image_list) #, add_border=True)
    

def visualize_all_bestfit_images(layers):
    return [visualize_bestfit_image(layer) for layer in layers]





if __name__ == '__main__':
    
    ################################################    
    # parse args
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--loss', default=False, action='store_true')
    parser.add_argument('--activations', default=False, action='store_true')
    parser.add_argument('--timelapse', default=False, action='store_true')
    parser.add_argument('--weights', default=False, action='store_true')
    parser.add_argument('--bestfit', default=False, action='store_true')
    args = parser.parse_args()

    # plot loss
    # note: needs to be done before loading dataset or model or it crashes (OOM).
    # don't know why...
    if args.loss or args.all:
        print('Plotting loss...')
        visualize_loss(args.dir)

        
    # load data, model, and checkpoint
    print('Loading dataset...')
    data = get_dataset(args.data)
    sample_indexes = np.random.choice(data.test.images.shape[0], args.examples, replace=False)
    example_images = data.test.images[sample_indexes, :]
    print('Loading model and checkpoint..')
    sess = reload_session(args.dir)        


    ################################################    
    # prep dirs
    
    # example images and activations
    for i in range(args.examples):
        example = example_images[i]
        sample_num = sample_indexes[i]
        example_path = os.path.join(args.dir, 'images', 'activations', 'example' + str(i))
        if not os.path.exists(example_path):
            os.makedirs(example_path)
        image_path = os.path.join(example_path, 'original_' + str(sample_num) + '.png')
        cv2.imwrite(image_path, example * 255.0)
        
    # weights and bestfit
    for d in ['weights', 'bestfit']:
        p = os.path.join(args.dir, 'images', d)
        if not os.path.exists(p):
            os.makedirs(p)


    ################################################
    # generate visualizations!
    
    # activations
    # TODO: add timelapse for each checkpoint
    if args.activations or args.all:
        print('Generating activations visualization...')
        layers = tf.get_collection('layers')
        for n in range(args.examples):
            example = example_images[n]
            example_path = os.path.join(args.dir, 'images', 'activations', 'example' + str(n))
            results = visualize_all_activations(layers, example)
            for i in range(len(results)):
                image_path = os.path.join(example_path, 'activation_layer_' + str(i) + '.png')
                cv2.imwrite(image_path, results[i])


    # example timelapse
    if args.timelapse or args.all:
        print('Generating timelapse...')
        results = visualize_timelapse(args.dir, example_images)
        cv2.imwrite(os.path.join(args.dir, 'images', 'timelapse.png'), results)
        # reset session to most recent checkpoint
        sess = reload_session(args.dir)


    # weights
    # note, the only variables we are interested in are the conv weights, which have shape of length 4
    if args.weights or args.all:
        print('Generating weights visualization...')
        weight_vars = [v for v in tf.trainable_variables() if len(v.get_shape()) == 4]
        results = visualize_all_weights(weight_vars)
        for i in range(len(results)):
            weight_path = os.path.join(args.dir, 'images', 'weights', 'weights_' + str(i) + '.png')
            cv2.imwrite(weight_path, results[i])

        
    # best fit via gradient ascent
    if args.bestfit:
        print('Generating best fit images for each filter...') 
        layers = tf.get_collection('layers')
        i = 0
        for i in range(2):
        # for i in range(len(layers)):        
            print('Generating layer', i)
            sess = reload_session(args.dir)
            layer = tf.get_collection('layers')[i]
            results = visualize_bestfit_image(layer)
            img_path = os.path.join(args.dir, 'images', 'bestfit', 'bestfit_layer_' + str(i) + '.png')
            cv2.imwrite(img_path, results)

        
