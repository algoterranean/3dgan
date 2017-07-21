"""Useful training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pickle
import re

# from data import TFRecordsDataset
from ops.input import batch_slice



def default_training(train_op):
    """Trainining function that just runs an op (or list of ops)."""
    losses = collection_to_dict(tf.get_collection('losses'))

    def helper(sess, args):
        _, results = sess.run([train_op, losses])
        return results
    return helper


def average_gradients(tower_grads):
    """Average the gradients from all GPU towers.

    Args:
      tower_grads: List, tuples of (grad, var) values.

    Returns:
      A list containing the averaged gradients in (grad, var) form.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # print('GRAD', grad)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def init_optimizer(args):
    """Helper function to initialize an optimizer from given arguments.

    Args:
      args: Argparse structure.

    Returns:
      An initialized optimizer.
    """
    with tf.variable_scope('optimizers'):
        if args.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(args.lr,
                                             decay= rgs.decay,
                                             momentum=args.momentum,
                                             centered=args.centered)
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
            return tf.train.MomentumOptimizer(args.lr,
                                              args.momentum)
        elif args.optimizer == 'adam':
            return tf.train.AdamOptimizer(args.lr,
                                          args.beta1,
                                          args.beta2)
        elif args.optimizer == 'ftrl':
            return tf.train.FtrlOptimizer(args.lr)
