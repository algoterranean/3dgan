import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # only log errors
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)   # only log errors
import numpy as np
import sys
import random
import argparse
import uuid
import pickle 
import h5py
import cv2
import time
from sys import stdout
from os import path

from msssim import MultiScaleSSIM, tf_ssim, tf_ms_ssim
from util import *
from data import get_dataset
from models.cnn import cnn
from models.gan import gan
from models.vae import vae



if __name__ == '__main__':
    
    # command line arguments
    ######################################################################
    parser = argparse.ArgumentParser(description='Autoencoder training harness.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         epilog="""Example:
                                                   python train.py --model gan 
                                                                   --data floorplans
                                                                   --epochs 100
                                                                   --batch_size 192
                                                                   --n_gpus 2
                                                                   --dir workspace/gan/run1""")
    parser._action_groups.pop()
    model_args = parser.add_argument_group('Model')
    data_args = parser.add_argument_group('Data')
    optimizer_args = parser.add_argument_group('Optimizer')    
    train_args = parser.add_argument_group('Training')
    misc_args = parser.add_argument_group('Miscellaneous')
    
    # misc settings
    add = misc_args.add_argument
    add('--seed',
            type=int,
            help="Useful for debugging. Randomized each execution if not set.")
    add('--n_gpus',
            type=int,
            default=1,
            help="""Number of GPUs to use for simultaneous training. Model will be 
                    duplicated on each device and results averaged on CPU.""")
    add('--profile',
            default=False,
            action='store_true',
            help="""Enables runtime metadata collection during training that is 
                    viewable in TensorBoard.""")
    # training settings
    add = train_args.add_argument
    add('--epochs',
            type=int,
            default=3,
            help="Number of epochs to train for during this run.")
    add('--batch_size',
            type=int,
            default=256,
            help="Batch size to use, per device.")
    add('--examples',
            type=int,
            default=64,
            help="""Number of examples to generate when sampling from generative models 
                    (if supported). Note, this must be a power of 2.""")
    add('--dir',
            type=str,
            default='workspace/{}'.format(uuid.uuid4()),
            help="""Location to store checkpoints, logs, etc. If this location is populated 
                    by a previous run then training will be continued from last checkpoint.""")
    add('--summary_freq',
            type=int,
            default=120,
            help="Run summary op every n seconds.")
    add('--checkpoint_freq',
            type=int,
            default=600,
            help="Save checkpoint every n seconds.")
    add('--n_disc_train',
            type=int,
            default=5,
            help="""Number of times to train discriminator before training generator 
                    (if applicable).""")
    # optimizer settings
    add = optimizer_args.add_argument
    add('--optimizer',
            type=lambda s: s.lower(),
            default='rmsprop',
            help="Optimizer to use during training.")
    add('--lr',
            type=float,
            default=0.001,
            help="Learning rate of optimizer (if supported).")
    add('--loss',
            type=lambda s: s.lower(),
            default='l1',
            help="Loss function used by model during training (if supported).")
    add('--momentum',
            type=float,
            default=0.01,
            help="Momentum value used by optimizer (if supported).")
    add('--decay',
            type=float,
            default=0.9,
            help="Decay value used by optimizer (if supported).")
    add('--centered',
            default=False,
            action='store_true',
            help="Enables centering in RMSProp optimizer.")
    add('--beta1',
            type=float,
            default=0.9,
            help="Value for optimizer's beta_1 (if supported).")
    add('--beta2',
            type=float,
            default=0.999,
            help="Value for optimizer's beta_2 (if supported).")
    # model settings
    add = model_args.add_argument
    add('--model',
            type=lambda s: s.lower(),
            default='fc',
            help="Name of model to train.")
    add('--latent_size',
            type=int,
            default=200,
            help="""Size of middle 'z' (or latent) vector to use in autoencoder 
                    models (if supported).""")
    # data/pipeline settings
    add = data_args.add_argument
    add('--dataset',
            type=lambda s: s.lower(),
            default='floorplans',
            help="Name of dataset to use. Default: floorplans.")
    add('--resize',
            type=int,
            nargs=2,
            help="""Resize input images to size w x h. This argument, if specified, 
                    requires two values (width and height).""")
    add('--grayscale',
            default=False,
            action='store_true',
            help="Converts input images to grayscale.")
    
    args = parser.parse_args()


    
    # set up model, data, and training environment
    ######################################################################
    # set seed (useful for debugging purposes)
    if args.seed is None:
        args.seed = os.urandom(4)
    random.seed(args.seed)

    
    # init globals
    message('Parsing options...')
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # save the arguments to the graph
        with tf.variable_scope('args'):
            for a in vars(args):
                v = getattr(args, a)
                if v is None:
                    v = False
                tf.Variable(v, name=a, trainable=False)
                print('    {} = {}'.format(a, v))                


    # input pipeline
    message('Initializing input pipeline...')
    d = get_dataset(args.dataset)
    x = d.batch_tensor(args.batch_size * args.n_gpus, args.epochs)
    if args.dataset == 'floorplans':
        # convert from BGR to RGB
        x = tf.reverse(x, axis=[-1])
    if args.resize:
        message('    Resizing images to {}.'.format(args.resize))
        x = tf.image.resize_images(x, args.resize)
    if args.grayscale:
        message('    Converting images to grayscale.')
        x = tf.image.rgb_to_grayscale(x)
        
    # setup model
    message('Initializing model...')
    # models should return a 2-tuple (f, s) where f is a training
    # function that runs one step (or batch) of training and s is a
    # summary op containing all summaries to run.
    model_funcs = {'gan'  : gan,
                   'wgan' : gan,
                   'iwgan': gan,
                   'vae'  : vae,
                   'cnn'  : cnn}
    train_func, summary_op = model_funcs[args.model](x, args)
    # get losses in dict form for easy printing during training
    losses = collection_to_dict(tf.get_collection('losses'))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    
    # supervisor
    message('Initializing supervisor...')
    supervisor = tf.train.Supervisor(
        logdir = args.dir,
        init_op = init_op,
        summary_op = summary_op,
        global_step = global_step,
        save_summaries_secs = args.summary_freq,
        save_model_secs = args.checkpoint_freq)

    # profiling (optional)
    # requires adding libcupti.so.8.0 (or equivalent) to LD_LIBRARY_PATH.
    # (location is /cuda_dir/extras/CUPTI/lib64)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if args.profile else None
    run_metadata = tf.RunMetadata() if args.profile else None

    
    # training
    ######################################################################
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with supervisor.managed_session(config=session_config) as sess:
        message('Starting training...')            
        start_time = time.time()
        supervisor.loop(args.summary_freq,
                            status_tracker,
                            args=(sess, global_step, losses, start_time))
        while not supervisor.should_stop():
            train_func(sess, args)
            
    message('\nTraining complete! Elapsed time: {}s'.format(int(time.time() - start_time)))
