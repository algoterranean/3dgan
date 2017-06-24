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
from tqdm import tqdm, trange
from sys import stdout
from os import path

from util import *
from data import get_dataset
from models.cnn import cnn
from models.gan import gan
from models.vae import vae


class load_args_from_file(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # add '--' prefix to options if not set
        contents = values.read().split()
        for i in range(int(len(contents)/2)):
            if contents[i*2][0:2] != '--':
                contents[i*2] = '--' + contents[i*2]
        # parse
        data = parser.parse_args(contents, namespace=namespace)
        # set values, ignoring any --config option in file
        for k, v in vars(data).items():
            if v and k != option_string.strip('-'):
                setattr(namespace, k, v)
                

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
    add('--config',
            type=open,
            action=load_args_from_file,
            help="""Read in a file containing command arguments. Any additional command 
                    line arguments will overwrite those same options set in the config file.""")
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
            default='3',
            help="""Number of epochs to train for during this run. Use an integer to
                  denote the max number of epochs to train for, or `+n` for an 
                  additional n epochs from a saved checkpoint.""")
    add('--batch_size',
            type=int,
            default=256,
            help="Batch size to use, per device.")
    add('--epoch_size',
            type=int,
            default=-1,
            help="""Number of iterations to use per epoch. Defaults to using the 
                    entire dataset.""")
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
    add('--shuffle',
            default=True,
            action='store_true',
            help="""Set this to shuffle the dataset every epoch.""")
    add('--buffer_size',
            type=int,
            default=10000,
            help="""Size of the data buffer.""")
    add('--grayscale',
            default=False,
            action='store_true',
            help="Converts input images to grayscale.")
    add('--cache_dir',
            default=None,
            help="""Cache dataset to the directory specified. If not provided, 
                    will attempt to cache to memory.""")
    
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
        global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
        increment_global_epoch = tf.assign(global_epoch, global_epoch+1)
        # # save the arguments to the graph
        # with tf.variable_scope('args'):
        #     for a in vars(args):
        #         v = getattr(args, a)
        #         if v is None:
        #             v = False
        #         tf.Variable(v, name=a, trainable=False)
        #         print('    {} = {}'.format(a, v))


    # input pipeline
    message('Initializing input pipeline...')
    with tf.variable_scope('input_pipeline'):
        x, x_init, x_count = get_dataset(args)
        # x, x_count = get_dataset(args)
        if args.epoch_size <= 0:
            iter_per_epoch = int(x_count / (args.batch_size * args.n_gpus))
        else:
            iter_per_epoch = args.epoch_size

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

    
    # supervisor
    message('Initializing supervisor...')
    init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    sv = tf.train.Supervisor(logdir = args.dir,
                                 init_op = init_op,
                                 summary_op = None,
                                 global_step = global_step,
                                 save_model_secs = 0,
                                 saver = tf.train.Saver(max_to_keep=0, name='saver'))

    # profiling (optional)
    # requires adding libcupti.so.8.0 (or equivalent) to LD_LIBRARY_PATH.
    # (location is /cuda_dir/extras/CUPTI/lib64)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if args.profile else None
    run_metadata = tf.RunMetadata() if args.profile else None
    

        
    
    # training
    ######################################################################
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=session_config) as sess:
        # initialize
        start_time = time.time()
        save_path = os.path.join(args.dir, 'checkpoint')
        current_step = int(sess.run(global_step))
        current_epoch = int(sess.run(global_epoch))
        if args.epochs[0] == '+':
            max_epochs = current_epoch + int(args.epochs[1:])
        else:
            max_epochs = int(args.epochs)
        status = None

        sess.run(x_init)        

        # save model params before any training has been done
        if current_step == 0:
            message('Generating baseline summaries and checkpoint...')
            # sess.run(x_init)
            sv.saver.save(sess, save_path=save_path, global_step=global_step)
            sv.summary_computed(sess, sess.run(summary_op))
            
        message('Starting training...')
        for epoch in range(current_epoch, max_epochs):
            # sess.run(x_init)
            # -1 to save 1 batch for summaries at end
            pbar = tqdm(range(iter_per_epoch), desc='Epoch {:3d}'.format(epoch+1), unit='batch')
            
            for i in pbar:
                if sv.should_stop():
                    print('stopping1')
                    break
                else:
                    # train and display status
                    prev_status = status
                    status = train_func(sess, args)
                    pbar.set_postfix(format_for_terminal(status, prev_status))

                    # record 10 extra summaries (per epoch) in the first 3 epochs
                    if epoch < 3 and i % int((iter_per_epoch / 10)) == 0:
                        sv.summary_computed(sess, sess.run(summary_op))
                        
                    # # and record a summary half-way through each epoch after that
                    elif epoch >= 3 and i % int((iter_per_epoch / 2)) == 0:
                        sv.summary_computed(sess, sess.run(summary_op))
                        
            if sv.should_stop():
                print('stopping2')
                break

            sess.run(increment_global_epoch)
            # sess.run(tf.assign(global_epoch, global_epoch+1))
            current_epoch = int(sess.run(global_epoch))
            # print('completed epoch {}'.format(current_epoch))

            # generate summaries and checkpoint
            sv.summary_computed(sess, sess.run(summary_op))
            sv.saver.save(sess, save_path=save_path, global_step=global_epoch)
            # print('generated summaries and checkpoint')
            
    message('\nTraining complete! Elapsed time: {}s'.format(int(time.time() - start_time)))
