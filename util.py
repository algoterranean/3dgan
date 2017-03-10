import numpy as np
import tensorflow as tf
import os
import sys
import cv2
from data import Floorplans

# helper functions
def generate_example_row(data, tensor, xs, include_actual, sess, x):
    examples = sess.run(tensor, feed_dict={x: xs})
    montage = None
    for i, pred in enumerate(examples):
        if include_actual:
            # v = np.vstack((np.reshape(data.test.images[i], (28, 28)) * 255.0,
            #                np.reshape(pred, (28, 28)) * 255.0))
            # v = np.vstack((np.reshape(data.test.dataset[i], (64, 64, 3)) * 255.0,
            #                np.reshape(pred, (64, 64, 3)) * 255.0))
            gray_img = data.test.dataset[i]
            # gray_img = cv2.cvtColor(data.test.dataset[i], cv2.COLOR_BGR2GRAY)
            # v = np.vstack((gray_img * 255.0,
            #                np.reshape(pred, (64, 64)) * 255.0))
            v = np.vstack((gray_img * 255.0, pred * 255.0))
            # v = np.vstack((np.reshape(data.test.dataset[i], (64, 64)) * 255.0,
            #                np.reshape(pred, (64, 64)) * 255.0))
        else:
            # v = np.reshape(pred, (28, 28)) * 255.0
            # v = np.reshape(pred, (64, 64, 3)) * 255.0
            v = pred * 255.0
            # v = np.reshape(pred, (64, 64)) * 255.0

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
