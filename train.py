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
# local
from models.fc import simple_fc
from models.conv import simple_cnn
from models.chen import chen_cnn
from models.shared_cnn import shared_cnn
# from models import fc.simple_fc, simple_cnn, chen_cnn, shared_cnn
from msssim import MultiScaleSSIM, tf_ssim, tf_ms_ssim
from data import Floorplans
from util import *



# TODO move this stuff to functions
# TODO add interactive back in as the default, but write a summary to disk for later review (in case terminal session is lost)


# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--examples', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--layers', type=int, nargs='+', default=(256, 128, 64))
parser.add_argument('--seed', type=int, default=os.urandom(4))
parser.add_argument('--dataset', type=lambda s: s.lower(), default='mnist')
parser.add_argument('--dir', type=str, default='workspace/{}'.format(uuid.uuid4()))
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--interactive', default=False, action='store_true')
parser.add_argument('--model', type=lambda s: s.lower(), default='fc')
parser.add_argument('--grayscale', default=False, action='store_true')
parser.add_argument('--loss', type=lambda s: s.lower(), default='l1')
parser.add_argument('--optimizer', type=lambda s: s.lower(), default='rmsprop')
parser.add_argument('--momentum', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0.9)
parser.add_argument('--centered', default=False, action='store_true')
args = parser.parse_args()

# TODO: should keep the args that were passed in this time
if args.resume:
    # copy passed-in values
    args_copy = argparse.Namespace(**vars(args))
    # load previous args
    args = pickle.load(open(os.path.join(args.dir, 'settings'), 'rb'))
    # adjust for new passed in values
    args.resume = True
    args.epochs = args_copy.epochs
    print('Args restored')
    
    

# seed
print('Setting seed to', args.seed)
random.seed(args.seed)

# session and global vars
sess = tf.Session()

# variables to track training progress. useful when resuming training from disk.
with tf.variable_scope('global_vars'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_epoch = tf.Variable(0, name='global_epoch', trainable=False)
    global_batchsize = tf.Variable(args.batchsize, name='global_batchsize', trainable=False)


# input tensor for dataset.
# x_input is the tensor for our actual data
# x is the tensor to be passed to the model (that is, after processing of actual data)
# TODO: move this to Dataset class or something. dataset should be self-describing
assert args.dataset in ['mnist', 'floorplans'], "Invalid dataset name '{}'".format(args.dataset)
with tf.variable_scope('inputs'):
    if args.dataset == 'mnist':
        x_input = tf.placeholder("float", [None, 784], name='x_input')
        x = tf.reshape(x_input, [-1, 28, 28, 1], name='x')
        # x = x - tf.reduce_mean(x)
    elif args.dataset == 'floorplans':
        x_input = tf.placeholder("float", [None, 64, 64, 3], name='x_input')
        x = tf.image.rgb_to_grayscale(x_input, name='x') if args.grayscale else tf.identity(x_input, name='x')
        # x = x - tf.reduce_mean(x)


assert args.model in ['fc', 'cnn', 'chencnn', 'sharedcnn'], "Invalid model name '{}'".format(args.model)
with tf.variable_scope('outputs'):
    # model    
    if args.model == 'fc':
        y_hat, model_summary_nodes = simple_fc(x, args.layers)
    elif args.model == 'cnn':
        y_hat, model_summary_nodes = simple_cnn(x, args.layers)
    elif args.model == 'chencnn':
        print('using chens cnn')
        y_hat, model_summary_nodes = chen_cnn(x)
    elif args.model == 'sharedcnn':
        y_hat, model_summary_nodes = shared_cnn(x)
    print('y_hat:', y_hat)
    y_hat = tf.identity(y_hat, name='y_hat')


# loss
with tf.variable_scope('loss_functions'):
    loss_functions = {'l1': tf.reduce_mean(tf.abs(x - y_hat), name='l1'),
                      'l2': tf.reduce_mean(tf.pow(x - y_hat, 2), name='l2'),
                      'rmse': tf.sqrt(tf.reduce_mean(tf.pow(x - y_hat, 2)), name='rmse'),
                      'mse': tf.reduce_mean(tf.pow(x - y_hat, 2), name='mse'),
                      'ssim': tf.subtract(1.0, tf_ssim(tf.image.rgb_to_grayscale(x), tf.image.rgb_to_grayscale(y_hat)), name='ssim'),
                      'crossentropy': -tf.reduce_sum(x * tf.log(y_hat), name='crossentropy')}
loss = loss_functions[args.loss]
print('using loss:', loss)

# optimizer
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
optimizer = optimizers[args.optimizer]
print('using optimizer:', optimizer)


# training step
train_step = optimizer.minimize(loss, global_step=global_step)
print('using train step:', train_step)


    
        
# workspace
log_files = prep_workspace(args.dir)


# session saving/loading
saver = tf.train.Saver(max_to_keep=None)
if not args.resume:
    pickle.dump(args, open(os.path.join(args.dir, 'settings'), 'wb'))
    tf.train.export_meta_graph(os.path.join(args.dir, 'model'))
    sess.run(tf.global_variables_initializer())
else:
    #saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))
    print('Model restored. Currently at step {} and epoch {} with batch size {}'.format(sess.run(global_step), sess.run(global_epoch), sess.run(global_batchsize)))


# tensorboard
montage = None
tb_writer = tf.summary.FileWriter(os.path.join(args.dir, 'logs'), graph=tf.get_default_graph())
# add tensorboard node on model output for generating examples
epoch_summary_nodes = tf.summary.merge([model_summary_nodes, tf.summary.image('model_output', y_hat * 255.0)])
batch_summary_nodes = tf.summary.merge([tf.summary.scalar('loss', loss),
                                        tf.summary.scalar('l1', loss_functions['l1']),
                                        tf.summary.scalar('l2', loss_functions['l2']),
                                        tf.summary.scalar('rmse', loss_functions['rmse']),
                                        tf.summary.scalar('crossentropy', loss_functions['crossentropy'])])
train_start_summary_nodes = None
train_end_summary_nodes = None


# print out schematic/visualization of model
total_params = visualize_parameters()
print('Total params: {}'.format(total_params))




# dataset
data = get_dataset(args.dataset)
# # example data for visualizing results/progress
sample_indexes = np.random.choice(data.test.images.shape[0], args.examples, replace=False)
print('Using example images', sample_indexes)
example_images = data.test.images[sample_indexes, :]

for r in example_images:
    print('example image:', r.min(), r.max())

# training!
print('Starting training')
start_epoch = sess.run(global_epoch) + 1
n_trbatches = int(data.train.num_examples/args.batchsize)
iterations_completed = sess.run(global_step) * args.batchsize

example_num = 0

# for each epoch...
for epoch in range(start_epoch, args.epochs+start_epoch):
    epoch_start_time = time.time()
    training_start_time = time.time()

    # TODO: shuffle data every epoch!
    
    # perform training
    total_train_loss = 0.0
    for i in range(n_trbatches):
        xs, ys = data.train.next_batch(args.batchsize)
        # print('xs:', xs)
        # print('ys:', ys)
        # print('batch:', xs.shape, ys.shape)
        _, l = sess.run([train_step, loss], feed_dict={x_input: xs})
        total_train_loss += l
        iterations_completed += args.batchsize
        log_files['train_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, l))
        if args.interactive:
            print_progress(epoch, iterations_completed, data.train.num_examples, l)
        if batch_summary_nodes is not None:
            summary_result = sess.run(batch_summary_nodes, feed_dict={x_input: xs})
            tb_writer.add_summary(summary_result, iterations_completed)
        if i % int(n_trbatches / 4) == 0:
            print('Adding images')
            # update tensorboard nodes
            if epoch_summary_nodes is not None:
                print('adding images 2')
                results = sess.run(y_hat, feed_dict={x_input: example_images})
                # print('results:', results)
                for r in results:
                    # print(r)
                    image_path = os.path.join(args.dir, 'images', 'example_{}.png'.format(example_num))
                    cv2.imwrite(image_path, r * 255)
                    print(example_num, r.shape, r.max())
                    example_num += 1
                    
                summary_result = sess.run(epoch_summary_nodes, feed_dict={x_input: example_images})
                tb_writer.add_summary(summary_result, i * args.batchsize + data.train.num_examples * (epoch-1))
            
        
    avg_train_loss = total_train_loss/n_trbatches
    training_end_time = time.time()
    if not args.interactive:
        print('Epoch {}: Train loss ({:.5f}), elapsed time {}'.format(epoch, avg_train_loss, training_end_time-training_start_time))
        
    # # perform validation
    # validation_start_time = time.time()
    # n_valbatches = int(data.validation.num_examples/args.batchsize)
    # total_validation_loss = 0.0
    # for i in range(n_valbatches):
    #     xs, ys = data.validation.next_batch(args.batchsize)
    #     total_validation_loss += sess.run(loss, feed_dict={x_input: xs})
    # validation_end_time = time.time()
    # avg_validation_loss = total_validation_loss/n_valbatches
    # log_files['validate_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, avg_validation_loss))
    # if args.interactive:
    #     sys.stdout.write(', validation: {:.4f}'.format(avg_validation_loss))
    #     sys.stdout.write('\r\n')
    # else:
    #     print('Epoch {}: Validation loss ({:.5f}), elapsed time {}'.format(epoch, avg_validation_loss, validation_end_time - validation_start_time))


    # update tensorboard nodes
    if epoch_summary_nodes is not None:
        summary_result = sess.run(epoch_summary_nodes, feed_dict={x_input: example_images})
        tb_writer.add_summary(summary_result, epoch)
        
        
    # snapshot
    if args.interactive:
        sys.stdout.write('Writing snapshot to disk...')
    else:
        print('Writing snapshot to disk...')
    chkfile = os.path.join(args.dir, 'checkpoints', 'epoch_{:03d}.ckpt'.format(epoch))
    saver.save(sess, chkfile, global_step=global_step)
    if args.interactive:
        sys.stdout.write('complete!\r\n')
        sys.stdout.flush()
    epoch_end_time = time.time()
    print('Total elapsed epoch time: {}'.format(epoch_end_time - epoch_start_time))

    # keep track of current epoch
    sess.run(global_epoch.assign(epoch+1))
    

    
# training completed!
print('Training completed')



# perform test
print('Starting testing')    
n_testbatches = int(data.test.num_examples/args.batchsize)
total_test_loss = 0.0
for i in range(n_testbatches):
    xs, ys = data.test.next_batch(args.batchsize)
    l = sess.run(loss, feed_dict={x_input: xs})
    total_test_loss += l
    if args.interactive:
        sys.stdout.write('\r')
        sys.stdout.write('test: {:.4f}'.format(l))
        sys.stdout.flush()
avg_test_loss = total_test_loss/n_testbatches
log_files['test_loss'].write('{:05d},{:.5f}\n'.format(iterations_completed, avg_test_loss))
if args.interactive:
    sys.stdout.write('\r\n')
else:
    print('Test loss: {:.5f}'.format(avg_test_loss))

    

# close down log files
for key in log_files:
    log_files[key].close()

