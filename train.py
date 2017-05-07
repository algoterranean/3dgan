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




def init_session(seed):
    random.seed(seed)
    return tf.Session()


def init_globals(sess):
    with sess.as_default():
        # variables to track training progress. useful when resuming training from disk.
        with tf.variable_scope('global_vars'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            global_epoch = tf.Variable(0, name='global_epoch', trainable=False)
            global_batchsize = tf.Variable(args.batchsize, name='global_batchsize', trainable=False)
    return global_step, global_epoch, global_batchsize



def init_input(sess, dataset, grayscale=False):
    with sess.as_default():
        # input tensor for dataset.
        # x_input is the tensor for our actual data
        # x is the tensor to be passed to the model (that is, after processing of actual data)
        # TODO: move this to Dataset class or something. dataset should be self-describing
        with tf.variable_scope('inputs'):
            if dataset == 'mnist':
                x_input = tf.placeholder("float", [None, 784], name='x_input')
                x = tf.reshape(x_input, [-1, 28, 28, 1], name='x')
                # x = x - tf.reduce_mean(x)
            elif dataset == 'floorplans':
                x_input = tf.placeholder("float", [None, 64, 64, 3], name='x_input')
                # x = tf.image.rgb_to_grayscale(x_input, name='x') if args.grayscale else tf.identity(x_input, name='x')
                x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_input)
                x = tf.identity(x, name='x')
                # tf.image.random_flip_left_right(img), images)
                # x = tf.image.per_image_standardization(x_input)


                # x = x - tf.reduce_mean(x)

            # x = tf.image.per_image_standardization(x)

        tf.add_to_collection('inputs', x)
        tf.add_to_collection('inputs', x_input)
    return x, x_input


def init_model(sess, optimizer, x, args): #, optimizer, global_step):
    with sess.as_default():
        if args.model == 'vae':
            return VAE(optimizer, x) #, optimizer, global_step)
        elif args.model == 'gan':
            return GAN(x, args.batchsize)
        elif args.model == 'cnn':
            return SimpleCNN(optimizer, x, args.layers)
        elif args.model == 'chen':
            return ChenCNN(optimizer, x)
        elif args.model == 'fc':
            return SimpleFC(optimizer, x, args.layers)
        elif args.model == 'vaegan':
            return VAEGAN(x)

        
            # # model    
            # if args.model == 'fc':
            #     model = SimpleFC(x, args.layers)
            # elif model_name == 'cnn':
            #     model = SimpleCNN(x, args.layers)            
            # elif model_name == 'chencnn':
            #     model = ChenCNN(x)
            # elif model_name == 'vae':
            #     model = VAE(x)
    # return model

    #         y_hat = tf.identity(model.output, name='y_hat')
    #     model_summary_nodes = tf.summary.merge([model.summary_nodes, tf.summary.image('model_output', y_hat * 255.0)])
        
    # return y_hat, model_summary_nodes


def init_loss(sess):
    with sess.as_default():
        with tf.variable_scope('loss_functions'):
            loss_functions = {'l1': tf.reduce_mean(tf.abs(x - y_hat), name='l1'),
                      'l2': tf.reduce_mean(tf.pow(x - y_hat, 2), name='l2'),
                      'rmse': tf.sqrt(tf.reduce_mean(tf.pow(x - y_hat, 2)), name='rmse'),
                      'mse': tf.reduce_mean(tf.pow(x - y_hat, 2), name='mse'),
                      # 'ssim': tf.subtract(1.0, tf_ssim(tf.image.rgb_to_grayscale(x), tf.image.rgb_to_grayscale(y_hat)), name='ssim'),
                      'crossentropy': -tf.reduce_sum(x * tf.log(y_hat), name='crossentropy')}
    return loss_functions


def init_optimizer(args):
    with tf.variable_scope('optimizers'):
        optimizers = {'rmsprop': tf.train.RMSPropOptimizer(args.lr, decay=args.decay, momentum=args.momentum, centered=args.centered),
                          'adadelta': tf.train.AdadeltaOptimizer(args.lr),
                          'gd': tf.train.GradientDescentOptimizer(args.lr),
                          'adagrad': tf.train.AdagradOptimizer(args.lr),
                          'momentum': tf.train.MomentumOptimizer(args.lr, args.momentum),
                          'adam': tf.train.AdamOptimizer(args.lr),
                          'ftrl': tf.train.FtrlOptimizer(args.lr),
                          'pgd': tf.train.ProximalGradientDescentOptimizer(args.lr),
                          'padagrad': tf.train.ProximalAdagradOptimizer(args.lr)}
    return optimizers[args.optimizer]


def init_tensorboard(sess, dir): #, model_summary): #, loss_functions):
    with sess.as_default():
        tb_writer = tf.summary.FileWriter(path.join(dir, 'logs'), graph=tf.get_default_graph())
        # epoch_summary_nodes = tf.summary.merge([model_summary])
        epoch_summary_nodes = tf.summary.merge(tf.get_collection('epoch_summaries'))
        batch_summary_nodes = tf.summary.merge(tf.get_collection('batch_summaries'))
        # batch_summary_nodes = tf.summary.merge([tf.summary.scalar(k, v) for k, v in loss_functions.items()])
        # batch_summary_nodes = tf.summary.merge([])
        train_start_summary_nodes = None
        train_end_summary_nodes = None
    return tb_writer, batch_summary_nodes, epoch_summary_nodes, train_start_summary_nodes, train_end_summary_nodes



if __name__ == '__main__':
    
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layers', type=int, nargs='+', default=(256, 128, 64))
    parser.add_argument('--seed', type=int, default=os.urandom(4))
    parser.add_argument('--dataset', type=lambda s: s.lower(), default='mnist')
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--dir', type=str, default='workspace/{}'.format(uuid.uuid4()))
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--model', type=lambda s: s.lower(), default='fc')
    parser.add_argument('--grayscale', default=False, action='store_true')
    parser.add_argument('--loss', type=lambda s: s.lower(), default='l1')
    parser.add_argument('--optimizer', type=lambda s: s.lower(), default='rmsprop')
    parser.add_argument('--momentum', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--centered', default=False, action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
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
    # init session, seed
    sess = init_session(args.seed)

    # init globals
    global_step, global_epoch, global_batchsize = init_globals(sess)

    # optimizer
    optimizer = init_optimizer(args)

    # setup input nodes (including any image filters)
    x, x_input = init_input(sess, args.dataset, grayscale=args.grayscale)

    # setup model
    with tf.variable_scope('model'):    
        model = init_model(sess, optimizer, x, args)
        # y_hat = tf.identity(model.decoder, name='y_hat')
    # model_summary_nodes = tf.summary.merge([tf.summary.image('decoder', y_hat)])
    # model_summary_nodes = tf.summary.merge([model.summary_nodes, tf.summary.image('model_output', y_hat * 255.0)])

    # loss
    # loss_functions = init_loss(sess)
    # loss = loss_functions[args.loss]

    loss = model.loss
    # train_op = optimizer.minimize(model.loss)

    # if args.model == 'vaegan':
    #     d_optim = optimizer.minimize(model._discriminator_loss)
    #     g_optim = loss
        
    # workspace
    log_files = prep_workspace(args.dir)

    # tensorboard
    tb_writer, batch_summary_nodes, epoch_summary_nodes, \
      train_start_summary_nodes, train_end_summary_nodes = init_tensorboard(sess, args.dir) #, model_summary_nodes) #, loss_functions)

    # session saving/loading
    saver = tf.train.Saver(max_to_keep=None)
    sess.run(tf.global_variables_initializer())
    if args.resume:
        saver.restore(sess, tf.train.latest_checkpoint(path.join(args.dir, 'checkpoints')))
        print('Model restored. Currently at step {} and epoch {} with batch size {}'.format(sess.run(global_step), sess.run(global_epoch), sess.run(global_batchsize)))
    else:
        save_settings(sess, args)

    # print out schematic/visualization of model
    total_params = visualize_parameters()
    print('Total params: {}'.format(total_params))

    # dataset
    debug('Loading dataset...')
    data = get_dataset(args.dataset)
    # example data for visualizing results/progress
    sample_indexes = np.random.choice(data.test.images.shape[0], args.batchsize, replace=False)
    example_images = data.test.images[sample_indexes, :]

    ######################################################################
    # training!
    debug('Starting training')
    start_epoch = sess.run(global_epoch) + 1
    n_trbatches = int(data.train.num_examples/args.batchsize)
    n_valbatches = int(data.validation.num_examples/args.batchsize)
    n_testbatches = int(data.test.num_examples/args.batchsize)
    iterations_completed = sess.run(global_step) * args.batchsize



    # for each epoch...
    for epoch in range(start_epoch, args.epochs+start_epoch):
        epoch_start_time = time.time()
        if args.shuffle:
            data.train.shuffle()        

        # perform training
        model.train(epoch, x_input, data, args.batchsize, tb_writer, batch_summary_nodes, epoch_summary_nodes) #, tb_writer, batch_summary_nodes)
        
        # for i in range(n_trbatches):
        #     # run train step
        #     xs, ys = data.train.next_batch(args.batchsize)
        #     if args.model == 'cnn':
        #         _, l = sess.run([train_op, loss], feed_dict={x_input: xs})
        #         print_progress(epoch, args.batchsize * (i+1), data.train.num_examples, epoch_start_time, l)
        #     elif args.model == 'vaegan':
        #         _, dl = sess.run([d_optim], feed_dict={x_input: xs})
        #         sampled_mu = np.random.normal(0, 1.0, (256, 512))
        #         _, gl = sess.run([g_optim], feed_dict={model._latent: sampled_mu})
        #         _, gl = sess.run([g_optim], feed_dict={model._latent: sampled_mu})
        #         print_progress(epoch, args.batchsize * (i+1), data.train.num_examples, epoch_start_time, dl, gl, ll)

        #     else:
        #         _, l, gl, ll = sess.run([train_op, loss, model.generated_loss, model.latent_loss], feed_dict={x_input: xs})
        #         print_progress(epoch, args.batchsize * (i+1), data.train.num_examples, epoch_start_time, l, gl, ll)


        
            # iterations_completed += args.batchsize
        
            # # log and print progress
            # log_files['train_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, l))
            
        
            # # run batch summary nodes
            # if batch_summary_nodes is not None:
            #     summary_result = sess.run(batch_summary_nodes, feed_dict={x_input: xs})
            #     tb_writer.add_summary(summary_result, iterations_completed)

            
        # # perform validation
        # results = fold(sess, x_input, [loss], data.validation, args.batchsize, n_valbatches)
        # # log and print progress
        # log_files['validate_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, results[0]))
        # stdout.write(', validation: {:.4f}\r\n'.format(results[0]))

        # # # run epoch summary nodes
        # if epoch_summary_nodes is not None:
        #     summary_result = sess.run(epoch_summary_nodes, feed_dict={x_input: example_images})
        #     tb_writer.add_summary(summary_result, epoch)
            
        # snapshot
        stdout.write('\tWriting snapshot to disk...')
        chkfile = path.join(args.dir, 'checkpoints', 'epoch_{:03d}.ckpt'.format(epoch))
        saver.save(sess, chkfile, global_step=global_step)
        stdout.write('complete!\r\n')
        stdout.flush()

        

        # # perform sampling
        # model.sample(sess)

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


