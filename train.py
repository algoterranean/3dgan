# stdlib/external
import tensorflow as tf
import numpy as np
import sys
import random
import argparse
import os
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



def init_input(sess, args): #dataset, grayscale=False):
    with sess.as_default():
        # input tensor for dataset.
        # x_input is the tensor for our actual data
        # x is the tensor to be passed to the model (that is, after processing of actual data)
        # TODO: move this to Dataset class or something. dataset should be self-describing
        with tf.variable_scope('inputs'):
            if args.dataset == 'mnist':
                x_input = tf.placeholder("float", [args.batch_size * args.n_gpus, 784], name='x_input')
                x = tf.reshape(x_input, [-1, 28, 28, 1], name='x')                
                # x_input = tf.placeholder("float", [None, 784], name='x_input')
            elif args.dataset == 'floorplans':
                x_input = tf.placeholder("float", [args.batch_size * args.n_gpus, 64, 64, 3], name='x_input')
                # x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_input)
                x = tf.identity(x_input, name='x')
                
                # x_input = tf.placeholder("float", [None, 64, 64, 3], name='x_input')
                # x = tf.image.rgb_to_grayscale(x_input, name='x') if args.grayscale else tf.identity(x_input, name='x')
                # tf.image.random_flip_left_right(img), images)
                # x = tf.image.per_image_standardization(x_input)
        tf.add_to_collection('inputs', x)
        tf.add_to_collection('inputs', x_input)
    return x, x_input




if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int, default=3)
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--examples',    type=int, default=10)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--layers',      type=int, nargs='+', default=(256, 128, 64))
    parser.add_argument('--seed',        type=int, default=os.urandom(4))
    parser.add_argument('--dataset',     type=lambda s: s.lower(), default='mnist')
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
    args = parser.parse_args()

    # TODO: should keep the args that were passed in this time, but not all of them?
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

    # setup workspace
    log_files = prep_workspace(args.dir)        
    
    # init session
    random.seed(args.seed)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) #, log_device_placement=True))

    # init globals
    with tf.device('/cpu:0'):
        with tf.variable_scope('global_vars'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            global_epoch = tf.Variable(0, name='global_epoch', trainable=False)
            global_batch_size = tf.Variable(args.batch_size, name='global_batch_size', trainable=False)    

    # setup input nodes (including any image filters)
    x, x_input = init_input(sess, args)

    # setup model
    
    with tf.variable_scope('model'):
        if args.model == 'gan':
            model = GAN(x, global_step, args)
    train_op = model.train_op
    losses = collection_to_dict(tf.get_collection('losses'))
    # losses = collection_to_dict(tf.get_collection('losses', scope='model'))    
    batch_summaries = tf.summary.merge_all() #key='batch')
    epoch_summaries = tf.summary.merge_all(key='epoch')
    # summaries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='model'))

    # setup tensorboard
    tb_writer = tf.summary.FileWriter(path.join(args.dir, 'logs'), graph=tf.get_default_graph())

    # session saving/loading
    saver = tf.train.Saver(max_to_keep=None)
    sess.run(tf.global_variables_initializer())
    if args.resume:
        saver.restore(sess, tf.train.latest_checkpoint(path.join(args.dir, 'checkpoints')))
        print('Model restored. Currently at step {} and epoch {} with batch size {}'.format(sess.run(global_step), sess.run(global_epoch), sess.run(global_batch_size)))
    else:
        save_settings(sess, args)

    # print out schematic/visualization of model
    total_params = visualize_parameters()
    print('Total params: {}'.format(total_params))

    # dataset
    debug('Loading dataset...')
    data = get_dataset(args.dataset)
    # example data for visualizing results/progress
    sample_indexes = np.random.choice(data.test.images.shape[0], args.batch_size, replace=False)
    example_images = data.test.images[sample_indexes, :]

    
    ######################################################################
    # training!
    
    debug('Starting training')
    start_epoch = sess.run(global_epoch) + 1
    n_trbatches = int(data.train.num_examples/(args.batch_size*args.n_gpus))
    # n_valbatches = int(data.validation.num_examples/args.batch_size)
    # n_testbatches = int(data.test.num_examples/args.batch_size)
    iterations_completed = sess.run(global_step) * args.batch_size

    # TODO add to args
    summary_freq = int(n_trbatches / 20)

    # for each epoch...
    for epoch in range(start_epoch, args.epochs+start_epoch):
        epoch_start_time = time.time()
        if args.shuffle:
            data.train.shuffle()

        # perform training
        for i in range(n_trbatches):
            xs, ys = data.train.next_batch(args.batch_size * args.n_gpus)
            # xs = []
            # for g in range(args.n_gpus):
            #     x_batch, y_batch = data.train.next_batch(args.batch_size)    
            #     xs.append(x_batch)
            
            _, l = sess.run([train_op, losses], {x_input: xs})
            print_progress(epoch, args.n_gpus*args.batch_size*(i+1), data.train.num_examples, epoch_start_time, l)

            # batch_summaries
            if i % summary_freq == 0:
                results = sess.run(model.summary_op, {x_input: xs})
                # results = sess.run(batch_summaries, {x_input: xs})
                tb_writer.add_summary(results, args.batch_size*(i+1) + (epoch-1)*n_trbatches*args.batch_size)

        # # epoch summaries
        # results = sess.run(epoch_summaries, {x_input: xs})
        # tb_writer.add_summary(results, args.batch_size*(i+1) + (epoch-1)*n_trbatches*args.batch_size)
        # # print('\n')

            
        # # perform validation
        # results = fold(sess, x_input, losses, data.validation, args.batchsize, n_valbatches)
        # # log_files['validate_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, results[0]))
        # stdout.write(', validation: {:.4f}\r\n'.format(results[0]))

            
        # snapshot
        stdout.write('\tWriting snapshot to disk...')
        chkfile = path.join(args.dir, 'checkpoints', 'epoch_{:03d}.ckpt'.format(epoch))
        saver.save(sess, chkfile, global_step=global_step)
        stdout.write('complete!\r\n')
        stdout.flush()

        # keep track of current epoch in global vars
        sess.run(global_epoch.assign(epoch+1))

        
    # # training completed, perform test
    # debug('Training completed')
    # debug('Starting testing')
    # results = fold(sess, x_input, [loss], data.test, args.batchsize, n_testbatches)
    # # # log and print progress
    # # log_files['test_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, results[0]))
    # stdout.write('Test loss: {:.4f}\r\n'.format(results[0]))
    # stdout.flush()


