# stdlib/external
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import sys
import random
import argparse
# import os
import uuid
import pickle 
import h5py
import cv2
import time
from sys import stdout
from os import path
# local
# from models.fc import SimpleFC
from models.fc import SimpleFC
from models.cnn import SimpleCNN
from models.chen import ChenCNN
from models.vae import VAE
from models.gan import GAN
from models.vaegan import VAEGAN
from msssim import MultiScaleSSIM, tf_ssim, tf_ms_ssim
from data import Floorplans
from util import *



# def init_input(sess, args): #dataset, grayscale=False):
#     with sess.as_default():
#         # input tensor for dataset.
#         # x_input is the tensor for our actual data
#         # x is the tensor to be passed to the model (that is, after processing of actual data)
#         # TODO: move this to Dataset class or something. dataset should be self-describing
#         with tf.variable_scope('inputs'):
#             if args.dataset == 'mnist':
#                 x_input = tf.placeholder("float", [args.batch_size * args.n_gpus, 784], name='x_input')
#                 x = tf.reshape(x_input, [-1, 28, 28, 1], name='x')                
#                 # x_input = tf.placeholder("float", [None, 784], name='x_input')
#             elif args.dataset == 'floorplans':
#                 x_input = tf.placeholder("float", [args.batch_size * args.n_gpus, 64, 64, 3], name='x_input')
#                 # x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_input)
#                 x = tf.identity(x_input, name='x')
                
#                 # x_input = tf.placeholder("float", [None, 64, 64, 3], name='x_input')
#                 # x = tf.image.rgb_to_grayscale(x_input, name='x') if args.grayscale else tf.identity(x_input, name='x')
#                 # tf.image.random_flip_left_right(img), images)
#                 # x = tf.image.per_image_standardization(x_input)
#         tf.add_to_collection('inputs', x)
#         tf.add_to_collection('inputs', x_input)
#     return x, x_input



def read_and_decode(filename_queue):
    feature_def = {'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string)}
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feature_def)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([64*64*3])
    image = tf.reshape(image, [64, 64, 3])
    image = tf.cast(image, tf.float32) * (1.0 / 255.0)  # - 0.5
    return image


def inputs(batch_size, num_epochs):
    filename = os.path.join('data', 'floorplans.64.train.tfrecords')
    with tf.name_scope('input_queue'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image = read_and_decode(filename_queue)
        images = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=1000+3*batch_size, min_after_dequeue=1000)
    return images







if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int, default=3)
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--examples',    type=int, default=10)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--layers',      type=int, nargs='+', default=(256, 128, 64))
    parser.add_argument('--seed',        type=int, default=os.urandom(4))
    parser.add_argument('--dataset',     type=lambda s: s.lower(), default='floorplans')
    parser.add_argument('--shuffle',     default=False, action='store_true')
    parser.add_argument('--dir',         type=str, default='workspace/{}'.format(uuid.uuid4()))
    parser.add_argument('--resume',      default=False, action='store_true')
    parser.add_argument('--model',       type=lambda s: s.lower(), default='fc')
    parser.add_argument('--grayscale',   default=False, action='store_true')
    parser.add_argument('--loss',        type=lambda s: s.lower(), default='l1')
    parser.add_argument('--optimizer',   type=lambda s: s.lower(), default='rmsprop')
    parser.add_argument('--momentum',    type=float, default=0.01)
    parser.add_argument('--decay',       type=float, default=0.9)
    parser.add_argument('--centered',    default=False, action='store_true')
    parser.add_argument('--n_gpus',      type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=200)
    parser.add_argument('--debug_graph', default=False, action='store_true')
    args = parser.parse_args()

    # TODO: how to integrate this into the Supervisor? 
    # store these values in the graph?
    
    if args.resume:
        # copy passed-in values
        args_copy = argparse.Namespace(**vars(args))
        # load previous args
        args = pickle.load(open(path.join(args.dir, 'settings'), 'rb'))
        # adjust for new passed in values
        args.resume = True
        args.epochs = args_copy.epochs
        print('Args restored')
    

    ######################################################################

    # init session
    random.seed(args.seed)

    # init globals
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # save the arguments to the graph
        with tf.variable_scope('args'):
            for a in vars(args):
                v = getattr(args, a)
                tf.Variable(v, name=a, trainable=False)

    # setup input pipeline
    x = inputs(args.batch_size * args.n_gpus, args.epochs)

    # setup model
    debug('Initializing model...')
    if args.model == 'gan':
        model = GAN(x, global_step, args)
        
    train_op = model.train_op
    losses = collection_to_dict(tf.get_collection('losses'))
    
    # # losses = collection_to_dict(tf.get_collection('losses', scope='model'))    
    # batch_summaries = tf.summary.merge_all() #key='batch')
    # epoch_summaries = tf.summary.merge_all(key='epoch')
    # summaries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='model'))

    saver = tf.train.Saver(max_to_keep=None)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def print_loss(sess):
        l, step = sess.run([losses, global_step])
        print_progress(step, l)
        
    debug('Initializing supervisor...')
    supervisor = tf.train.Supervisor(logdir=os.path.join(args.dir, 'logs'),
                                         init_op=init_op,
                                         summary_op=model.summary_op,
                                         global_step=global_step,
                                         save_summaries_secs=30,
                                         save_model_secs=300)
    
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.loop(30, print_loss, (sess, ))
        start_time = time.time()
        debug('Starting training...')
        while not supervisor.should_stop():
            sess.run(train_op)
            

    print('')
    debug('Training complete! Elapsed time: {}s'.format(int(time.time() - start_time)))
    
    
            
