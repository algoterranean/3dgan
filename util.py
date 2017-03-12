import numpy as np
import tensorflow as tf
import os
import sys
import cv2
from data import Floorplans

# helper functions
def generate_example_row(data, tensor, xs, include_actual, sess, x_input, grayscale=False):
    examples = sess.run(tensor, feed_dict={x_input: xs})
    montage = None
    for i, pred in enumerate(examples):

        if include_actual:
            if grayscale:
                input_img = cv2.cvtColor(data.test.dataset[i], cv2.COLOR_BGR2GRAY)
                pred = np.squeeze(pred)
            else:
                input_img = data.test.dataset[i]
            # print('pred:', pred.shape, 'input_img:', input_img.shape)
            v = np.vstack((input_img * 255.0, pred * 255.0))
        else:
            if grayscale:
                pred = np.squeeze(pred)
            v = pred * 255.0
        montage = v if montage is None else np.hstack((montage, v))
    return montage


def print_progress(epoch, completed, total, loss):
    sys.stdout.write('\r')
    sys.stdout.write('Epoch {:03d}: {:05d}/{:05d}: {:.4f}'.format(epoch, completed, total, loss))
    sys.stdout.flush()


def get_dataset(name):
    print('Loading dataset...')
    if name == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        return input_data.read_data_sets("data/MNIST_data", one_hot=True)
    elif name == 'floorplan':
        return Floorplans()


def prep_workspace(dirname):
    subdirs = [os.path.join(dirname, "checkpoints"),
               os.path.join(dirname, "images"),
               os.path.join(dirname, "logs")]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in subdirs:
        if not os.path.exists(d):
            os.mkdir(d)
            
    return {'train_loss': open(os.path.join(dirname, "logs", "train_loss.csv"), 'a'),
            'validate_loss': open(os.path.join(dirname, "logs", "validate_loss.csv"), 'a'),
            'test_loss' : open(os.path.join(dirname, "logs", "test_loss.csv"), 'a')}


def plot_loss(image_dir):
    pass
