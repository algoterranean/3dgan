# stdlib/external
import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import shutil
import time
# local
from data import Floorplans



def print_progress(epoch, completed, total, loss, start_time):
    end_time = time.time()
    sys.stdout.write('\r')
    sys.stdout.write('Epoch {:03d}: {:05d}/{:05d}: {:.4f} ({:d}s)'.format(epoch, completed, total, loss, int(end_time - start_time)))
    sys.stdout.flush()


def get_dataset(name):
    if name == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        return input_data.read_data_sets("data/MNIST_data", one_hot=True)
    elif name == 'floorplans':
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


def visualize_parameters():
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params = 1
        for dim in shape:
            num_params *= dim.value
        total_params += num_params
        print('Variable name: {}, size: {}, shape: {}'.format(variable.name, num_params, variable.get_shape()))
    return total_params


OKBLUE = '\033[94m'
ENDC = '\033[0m'
BOLD = '\033[1m'

def debug(*args):
    if len(args) > 1:
        print(BOLD + OKBLUE + str(args[0]), *(args[1:]), ENDC)
    else:
        print(BOLD + OKBLUE + str(args[0]), ENDC)


    
