"""Implementation of useful summary ops."""

from __future__ import absolute_import, division, print_function
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope

import tensorflow as tf
import hem
from math import sqrt
from operator import itemgetter



def summarize_activations(scope=None, summarize_montages=True):
    with tf.variable_scope('activations'):
        for l in tf.get_collection('conv_layers', scope=scope):
            layer_name = hem.tensor_name(l)
            l = tf.transpose(l, [0, 2, 3, 1])
            tf.summary.histogram(layer_name, l)
            tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(l))
            tf.summary.scalar(layer_name + '/mean', tf.reduce_mean(l))
            if summarize_montages:
                montage(tf.transpose(l[0], [2, 0, 1]), name=layer_name + '/montage')
        for l in tf.get_collection('dense_layers', scope=scope):
            # l = tf.transpose(l, [0, 2, 3, 1])
            layer_name = hem.tensor_name(l)
            tf.summary.histogram(layer_name, l)
            tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(l))
            tf.summary.scalar(layer_name + '/mean', tf.reduce_mean(l))


def summarize_layers(scope, layers, montage=False):
    with tf.variable_scope(scope):
        # activations
        for l in layers:
            layer_name = hem.tensor_name(l)
            layer_name = '/'.join(layer_name.split('/')[0:2])
            l = tf.transpose(l, [0, 2, 3, 1])
            tf.summary.histogram(layer_name, l)
            tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(l))
            tf.summary.scalar(layer_name + '/mean', tf.reduce_mean(l))
            if montage:
                hem.montage(tf.transpose(l[0], [2, 0, 1]), name=layer_name + '/montage')

            
def summarize_losses():
    with tf.variable_scope('loss'):
        for l in tf.get_collection('losses'):
            tf.summary.scalar(hem.tensor_name(l), l)
            tf.summary.histogram(hem.tensor_name(l), l)

            
def summarize_weights_biases():
    with tf.variable_scope('weights'):
        for l in tf.get_collection('weights'):
            tf.summary.histogram(hem.tensor_name(l), l)
            tf.summary.scalar(hem.tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))
            # montage_summary(l, name=tensor_name(l) + '/montage')
    with tf.variable_scope('biases'):
        for l in tf.get_collection('biases'):
            tf.summary.histogram(hem.tensor_name(l), l)
            tf.summary.scalar(hem.tensor_name(l) + '/sparsity', tf.nn.zero_fraction(l))
    

def summarize_gradients(grads_and_vars, name=None):
    """Adds histogram summaries for input list.

    Args:
      grads_and_vars: List, a list of tuples in form (grad, var).e
      name: String, name to use for var scope.
    """
    with tf.name_scope(name, 'gradients', grads_and_vars):
        for g, v in grads_and_vars:
            n = hem.tensor_name(v) + '/gradient'
            tf.summary.histogram(n, g)
            tf.summary.scalar(n, tf.reduce_mean(g))


def summarize_collection(name, scope):
    """Add a scalar summary for every tensor in a collection."""
    collection = tf.get_collection(name, scope)
    for x in collection:
        tf.summary.scalar(hem.tensor_name(x), x)
    return collection



def summarize_moments(x, name, args):
    mean, var = tf.nn.moments(x, axes=[0])
    tf.summary.scalar(name + '/mean', tf.reduce_mean(mean))
    tf.summary.scalar(name + '/var', tf.reduce_mean(var))
    tf.summary.histogram(name + '/mean', mean)
    tf.summary.histogram(name + '/var', var)
    var = tf.expand_dims(var, axis=0)
    var = hem.colorize(var)
    tf.summary.image(name + '/var', var)


# def activation_summary(x, rows=0, cols=0, montage=True, name=None):
#     """Summarize activations of input tensor.

#     Args:
#       x: Tensor, the input tensor to summarize. 
#       rows: Integer, number of rows in montage (if applicable).
#       cols: Integer, number of columns in montage (if applicable).
#       montage: Boolean, whether to generate an image montage of this tensor.
#     """
#     n = tensor_name(x)
#     # n = re.sub('tower_[0-9]*/', '', x.op.name).split('/')[-1]
#     # with tf.name_scope(name, 'activations', [x]) as scope:
        
#     tf.summary.histogram('activations', x)
#     tf.summary.scalar('activations/sparsity', tf.nn.zero_fraction(x))
#     if montage:
#         montage_summary(tf.transpose(x[0], [2, 0, 1]), 'montage')
        
#         # tf.summary.histogram(n + '/activations/histogram', x)
#         # tf.summary.scalar(n + '/activations/sparsity', tf.nn.zero_fraction(x))
#         # if montage:
#         #     montage_summary(tf.transpose(x[0], [2, 0, 1]), name=n + '/activations')
            

def factorization(n):
    """Finds factors of n suitable for image montage.

    Args:
      n: Integer, value to factorize. Should not be prime!

    Returns: 
      A tuple of form (m, n) representing the number of rows and
      columns this number factorizes to.
    """
    for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0:  # and i > 1:
            return i, int(n/i)
    # raise ValueError("Invalid montage grid size of {}".format(n))


@add_arg_scope
def montage(x, height=0, width=0, colorize=False, num_examples=-1, name=None):
    """Generates a m x n image montage from the given tensor.

    If m or n is 0, attempts to infer the input shape at run time.  Note
    that this may not always be possible.

    Args:
      x: Tensor, images to combine in montage. 
      m: Integer, number of rows in montage grid.
      n: Integer, number of columns in montage grid.
      name: String, name of this summary.

    Returns:
      Summary image node containing the montage.
    """


    with tf.name_scope(name, 'montage', [x]) as scope:
        # restrict montage to specific size
        if num_examples == -1:
            num_examples = x.get_shape()[0]
        x = x[0:num_examples]
        # colorize if asked and using a single-channel. otherwise make sure to convert to NHWC
        if colorize and x.get_shape()[1] == 1:
            x = hem.colorize(x)
        else:
            if len(x.shape) == 4:
                x = tf.transpose(x, [0, 2, 3, 1])
        # figure out w/h dynamically if necessary
        if height == 0 or width == 0:
            height, width = factorization(x.get_shape()[0].value)
        # create montage
        images = tf.split(x, width, axis=0)
        images = tf.concat(images, axis=1)
        images = tf.unstack(images, height, axis=0)
        images = tf.concat(images, axis=1)
        # if this is a b/w images, add a third dimension
        if len(images.shape) < 3:
            images = tf.expand_dims(images, axis=2)
        # add a first dimension for expected TB format
        images = tf.expand_dims(images, axis=0)
        y = tf.summary.image(scope, images)
    return y


def histograms(*args):
    for x in args:
        name, tensor = x
        tf.summary.histogram(name, tensor)


def scalars(*args):
    for x in args:
        name, tensor = x
        tf.summary.scalar(name, tensor)


def add_basic_summaries(gradients=None):
    if gradients is not None:
        summarize_gradients(gradients)
    summarize_activations()
    summarize_losses()
    summarize_weights_biases()


def get_all_events(event_files):
    events = {'scalar': {}, 'image': {}, 'histo': {}, 'tensor': {}}
    # get all available tags
    for f in event_files:
        for e in tf.train.summary_iterator(f):
            # print(e.step, e.wall_time)
            for v in e.summary.value:
                if v.image.height and v.image.width:
                    if v.tag not in events['image']:
                        events['image'][v.tag] = []
                    events['image'][v.tag].append((e.step, e.wall_time, v.image))
                elif v.histo.bucket:
                    if v.tag not in events['histo']:
                        events['histo'][v.tag] = []
                elif v.tensor.dtype:
                    if v.tag not in events['tensor']:
                        events['tensor'][v.tag] = []
                else:
                    if v.tag not in events['scalar']:
                        events['scalar'][v.tag] = []
                    events['scalar'][v.tag].append((e.step, e.wall_time, v.simple_value))
    return events


def get_tag_values(event_type, tag, events):
    # print('Found {} entries for tag {}.'.format(len(vals), tag))
    vals = events[event_type][tag]
    # sort by wall clock time, then step
    vals.sort(key=itemgetter(1), reverse=True)
    vals.sort(key=itemgetter(0))
    # remove duplicate entries. favor the more recent wall clock time
    cleaned_vals = []
    last_step = -1
    for v in vals:
        if last_step == v[0]:
            continue
        last_step = v[0]
        cleaned_vals.append(v)
    return cleaned_vals