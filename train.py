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

from models.cnn import CNN
from models.vae import VAE
from models.gan import GAN
from msssim import MultiScaleSSIM, tf_ssim, tf_ms_ssim
from util import *



if __name__ == '__main__':
    
    # command line arguments
    ######################################################################
    parser = argparse.ArgumentParser(description='Autoencoder training harness.',
                                         epilog="""Example:
                                                   python train.py --model gan 
                                                                   --data floorplans
                                                                   --dir workspace/gan/run1
                                                                   --epochs 100
                                                                   --batch_size 192
                                                                   --n_gpus 2""")
    parser._action_groups.pop()
    model_args = parser.add_argument_group('Model')
    data_args = parser.add_argument_group('Data')
    optimizer_args = parser.add_argument_group('Optimizer')    
    train_args = parser.add_argument_group('Training')
    misc_args = parser.add_argument_group('Miscellaneous')
    # misc settings
    misc_args.add_argument('--seed', type=int, default=os.urandom(4),
                               help='Useful for debugging. Default: output of os.urandom(4).')
    misc_args.add_argument('--n_gpus', type=int, default=1,
                               help="""Number of GPUs to use for simultaneous training. 
                                       Model will be duplicated on each device and results 
                                       averaged on CPU. Default: 1""")
    misc_args.add_argument('--profile',     default=False, action='store_true',
                               help="""Enables runtime metadata collection during training 
                                       that is viewable in TensorBoard. Default: False.""")
    # training settings
    train_args.add_argument('--epochs', type=int, default=3,
                                help="""Number of epochs to train for during this run. 
                                        Default: 3.""")
    train_args.add_argument('--batch_size', type=int, default=256,
                                help='Batch size _per GPU_. Default: 256.')
    train_args.add_argument('--examples', type=int, default=64,
                                help="""Number of examples to generate when sampling from 
                                        generative models (if supported). Default: 10.
                                        NOTE: Must be a power of 2!""")
    train_args.add_argument('--dir', type=str, default='workspace/{}'.format(uuid.uuid4()),
                                help="""Location to store checkpoints, logs, etc. If this 
                                        location is populated by a previous run then training 
                                        will be continued from last checkpoint.""")
    train_args.add_argument('--summary_freq', type=int, default=30,
                                help="""Run summary op every n seconds.
                                        Default: 100.""")
    train_args.add_argument('--n_disc_train', type=int, default=5,
                                help="""Number of times to train discriminator before 
                                        training generator (if applicable).
                                        Default: 5. """)
    # optimizer settings
    optimizer_args.add_argument('--optimizer', type=lambda s: s.lower(), default='rmsprop',
                                    help='Optimizer to use during training. Default: rmsprop.')
    optimizer_args.add_argument('--lr', type=float, default=0.001,
                                    help="""Learning rate of optimizer (if supported). 
                                            Default: 0.001.""")
    optimizer_args.add_argument('--loss', type=lambda s: s.lower(), default='l1',
                                    help="""Loss function used by model during training 
                                            (if supported). Default: l1.""")
    optimizer_args.add_argument('--momentum', type=float, default=0.01,
                                    help="""Momentum value used by optimizer (if supported). 
                                            Default: 0.01.""")
    optimizer_args.add_argument('--decay', type=float, default=0.9,
                                    help="""Decay value used by optimizer (if supported). 
                                            Default: 0.9.""")
    optimizer_args.add_argument('--centered',    default=False, action='store_true',
                                    help="""Enables centering in RMSProp optimizer. 
                                            Default: False.""")
    optimizer_args.add_argument('--beta1', type=float, default=0.9,
                                    help="""Value for optimizer's beta_1 (if supported).
                                            Default: 0.9.""")
    optimizer_args.add_argument('--beta2', type=float, default=0.999,
                                    help="""Value for optimizer's beta_2 (if supported).
                                            Default: 0.999.""")
    # model settings
    model_args.add_argument('--model', type=lambda s: s.lower(), default='fc',
                                help='Name of model to train. Default: fc.')
    model_args.add_argument('--layers', type=int, nargs='+', default=(256, 128, 64),
                                help="""Ordered list of layer sizes to use for sequential 
                                        models that stack layers (if supported). 
                                        Default: 256 128 64.""")
    model_args.add_argument('--latent_size', type=int, default=200,
                                help="""Size of middle 'z' (or latent) vector to use in 
                                        autoencoder models (if supported). Default: 200.""")
    # data/pipeline settings
    data_args.add_argument('--dataset', type=lambda s: s.lower(), default='floorplans',
                               help='Name of dataset to use. Default: floorplans.')
    data_args.add_argument('--resize', type=int, nargs=2, default=[0, 0],
                               help="""Resize input images to size w x h. This argument, if 
                                     specified, requires two values (width and height).""")
    args = parser.parse_args()

    
    # set up model, data, and training environment
    ######################################################################
    # set seed (useful for debugging purposes)
    random.seed(args.seed)
    
    # init globals
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # save the arguments to the graph
        with tf.variable_scope('args'):
            for a in vars(args):
                v = getattr(args, a)
                tf.Variable(v, name=a, trainable=False)

    # setup input pipeline
    d = get_dataset(args.dataset)
    x = d.batch_tensor(args.batch_size * args.n_gpus, args.epochs)

    # x = tf.image.resize_images(x, [32, 32])

    # setup model
    debug('Initializing model...')
    if args.model in ['gan', 'wgan', 'iwgan']:
        model = GAN(x, args)
    elif args.model == 'vae':
        model = VAE(x, args)
    elif args.model == 'cnn':
        model = CNN(x, args)
        
    losses = collection_to_dict(tf.get_collection('losses'))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # supervisor
    debug('Initializing supervisor...')
    supervisor = tf.train.Supervisor(logdir=args.dir,
                                         init_op=init_op,
                                         summary_op=model.summary_op,
                                         global_step=global_step,
                                         save_summaries_secs=120,
                                         save_model_secs=600)

    # profiling (optional)
    # requires adding libcupti.so.8.0 to LD_LIBRARY_PATH.
    # (location is /cuda_dir/extras/CUPTI/lib64)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if args.profile else None
    run_metadata = tf.RunMetadata() if args.profile else None

    # training
    ######################################################################
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with supervisor.managed_session(config=session_config) as sess:
        start_time = time.time()
        supervisor.loop(args.summary_freq, status_tracker, args=(sess, global_step, losses, start_time))
        debug('Starting training...')

        while not supervisor.should_stop():
            model.train(sess, args)
            
    debug('\nTraining complete! Elapsed time: {}s'.format(int(time.time() - start_time)))
