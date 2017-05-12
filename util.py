# stdlib/external
import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import shutil
import time
import pickle 
# local
from data import Floorplans




def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            # else:
            #     print('grads/vars is none:', g, v)
        # print('GRAD COUNT', len(grads), v)
        if len(grads) > 0:
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


# def average_gradients(tower_grads):
#   """Calculate the average gradient for each shared variable across all towers.
#   Note that this function provides a synchronization point across all towers.
#   Args:
#     tower_grads: List of lists of (gradient, variable) tuples. The outer list
#       is over individual gradients. The inner list is over the gradient
#       calculation for each tower.
#   Returns:
#      List of pairs of (gradient, variable) where the gradient has been averaged
#      across all towers.
#   """
#   average_grads = []
#   for grad_and_vars in zip(*tower_grads):
#     # Note that each grad_and_vars looks like the following:
#     #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#     grads = []
#     for g, _ in grad_and_vars:
#       # Add 0 dimension to the gradients to represent the tower.
#       expanded_g = tf.expand_dims(g, 0)

#       # Append on a 'tower' dimension which we will average over below.
#       grads.append(expanded_g)

#     # Average over the 'tower' dimension.
#     grad = tf.concat(0, grads)
#     grad = tf.reduce_mean(grad, 0)

#     # Keep in mind that the Variables are redundant because they are shared
#     # across towers. So .. we will just return the first tower's pointer to
#     # the Variable.
#     v = grad_and_vars[0][1]
#     grad_and_var = (grad, v)
#     average_grads.append(grad_and_var)
#   return average_grads


def init_optimizer(args):
    with tf.variable_scope('optimizers'):
        if args.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(args.lr, decay=args.decay, momentum=args.momentum, centered=args.centered)
        elif args.optimizer == 'adadelta':
            return tf.train.AdadeltaOptimizer(args.lr)
        elif args.optimizer == 'adagrad':
            return tf.train.AdagradOptimizer(args.lr)
        elif args.optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer(args.lr)
        elif args.optimizer == 'pgd':
            tf.train.ProximalGradientDescentOptimizer(args.lr)
        elif args.optimizer == 'padagrad':
            return tf.train.ProximalAdagradOptimizer(args.lr)
        elif args.optimizer == 'momentum':            
            return tf.train.MomentumOptimizer(args.lr, args.momentum)
        elif args.optimizer == 'adam':
            return tf.train.AdamOptimizer(args.lr)
        elif args.optimizer == 'ftrl':
            return tf.train.FtrlOptimizer(args.lr)


# TODO: improve this
def collection_to_dict(collection):
    d = {}
    for c in collection:
        name = c.name.split('/')[-1]
        name = name.split(':')[0]
        d[name] = c
    return d
        

# def print_progress(epoch, completed, total, loss, gl, ll, start_time):

def print_progress(epoch, completed, total, start_time, loss_dict):
    end_time = time.time()
    s = ""
    for k, v in loss_dict.items():
        s += '{}: {:.4f}, '.format(k, v)
    sys.stdout.write('\rEpoch {:03d}: {:05d}/{:05d}: {} ({:d}s)'.format(epoch, completed, total, s[:-2], int(end_time - start_time)))
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

    print('\nInputs:')
    print('=' * 40)
    for c in tf.get_collection('inputs'):
        print('{}, shape: {}'.format(c.name, c.get_shape()))
        
    # for c in tf.get_default_graph().get_operations():
    #     if c.name.split('/')[0] == 'inputs':
    #         print(c)
        

    categories = {}
    for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'):
        cat = c.name.split('/')[1]
        if not cat in categories:
            categories[cat] = []
        categories[cat].append(c)

    print('\nModel parameters:')
    print('=' * 40)
    for k,v in categories.items():
        debug(k)
        for item in v:
            shape = item.get_shape()
            print('\t{}, shape: {}'.format(item.name.split('/')[-1], shape))


    categories = {}
    for c in tf.get_collection('layer'):
        cat = c.name.split('/')[1]
        if not cat in categories:
            categories[cat] = []
        categories[cat].append(c)

        

    print('\nModel layers:')
    print('=' * 40)
    for k, v in categories.items():
        debug(k)
        for item in v:
            shape = item.get_shape()
            print('\t{}, shape: {}'.format(item.name.split('/')[-1], shape))

        

    
    # for c in collection:
    #     print(c.name, c.get_shape())
    



    # collection = tf.get_collection('layer', scope='model/encoder')
    # # collection.sort(key=lambda x: x.name)

    # # c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder')
    # # for tensor in c:
    # #     print(tensor)
    # #     print(tensor.consumers())


    # c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/encoder')
    # for tensor in c:
    #     print(tensor)
    # print('=' * 40)        

    # # for variable in tf.get_collection(tf.GraphKeys.W
    # for tensor in collection:
    #     shape = tensor.get_shape()
    #     print('Layer ({}), type: {}, shape: {}'.format(tensor.name, tensor.dtype.name, shape))
    #     print('\tInputs:')
    #     for x in tensor.op.inputs:
    #         print('\t\t', x)
    #     print('\tOutputs:', tensor.op.outputs)
    
    #     # if shape is not None:
    #     #     num_params = 1
    #     #     for dim in shape:
    #     #         num_params *= dim.value
    #     #     total_params += num_params
    #     # print('Layer ({}): size: {}, shape: {}'.format(tensor.name, 0, shape))

    #     # print('\tOn {}'.format(tensor.device))
    #     # print('\tConsumers:', tensor.consumers())
        

    return total_params
    

        
    # for variable in tf.trainable_variables():
    #     shape = variable.get_shape()
    #     num_params = 1
    #     for dim in shape:
    #         num_params *= dim.value
    #     total_params += num_params
    #     print('Variable name: {}, size: {}, shape: {}'.format(variable.name, num_params, variable.get_shape()))
    # return total_params


OKBLUE = '\033[94m'
ENDC = '\033[0m'
BOLD = '\033[1m'

def debug(*args):
    if len(args) > 1:
        print(BOLD + OKBLUE + str(args[0]), args[1:], ENDC)
    else:
        print(BOLD + OKBLUE + str(args[0]), ENDC)


    
def fold(sess, x_input, ops, data, batch_size, num_batches):
    """Runs each op on the data in batches and returns the average value for each op."""
    start_time = time.time()
    total_values = [0.0 for x in ops]
    for i in range(num_batches):
        xs, ys = data.next_batch(batch_size)
        results = sess.run(ops, feed_dict={x_input: xs})
        for j in range(len(results)):
            total_values[j] += results[j]
    end_time = time.time()
    avg_values = [x/num_batches for x in total_values]
    return avg_values


def save_settings(sess, args):
    """Writes settings and model graph to disk."""
    with sess.as_default():
        pickle.dump(args, open(os.path.join(args.dir, 'settings'), 'wb'))
        tf.train.export_meta_graph(os.path.join(args.dir, 'model'))


        
def reload_session(dir, fn=None):
    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(dir, 'model'))
    if fn is None:
        chk_file = tf.train.latest_checkpoint(os.path.join(dir, 'checkpoints'))
    else:
        chk_file = fn
    saver.restore(sess, chk_file)
    return sess


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# usage: list(chunks(some_list, chunk_size)) ==> list of lists of that size
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def checkpoints(workspace_dir, only_recent=False):
    checkpoint_files = []
    for f in os.listdir(os.path.join(workspace_dir, 'checkpoints')):
        if f.endswith('.meta'):
            checkpoint_files.append(f[:f.rfind('.')])
    checkpoint_files.sort()
    if only_recent:
        return [checkpoint_files[-1]]
    return checkpoint_files
