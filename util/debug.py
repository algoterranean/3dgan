"""Useful functions for debugging and displaying information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
UNDERLINE = '\033[4m'
BOLD = '\033[1m'
HEADER = '\033[95m'


def format_for_terminal(results):
    formatted_results = results
    for k in results:
        formatted_results[k] = '{:3f}'.format(results[k])
    return formatted_results


def visualize_parameters():
    total_params = 0

    print('\nInputs:')
    print('=' * 40)
    for c in tf.get_collection('inputs'):
        print('{}, shape: {}'.format(c.name, c.get_shape()))

    categories = {}
    for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'):
        cat = c.name.split('/')[1]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(c)

    print('\nModel parameters:')
    print('=' * 40)
    for k, v in categories.items():
        message(k)
        for item in v:
            shape = item.get_shape()
            print('\t{}, shape: {}'.format(item.name.split('/')[-1], shape))

    categories = {}
    for c in tf.get_collection('layer'):
        cat = c.name.split('/')[1]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(c)

    print('\nModel layers:')
    print('=' * 40)
    for k, v in categories.items():
        message(k)
        for item in v:
            shape = item.get_shape()
            print('\t{}, shape: {}'.format(item.name.split('/')[-1], shape))

    return total_params


def message(*args):
    if len(args) > 1:
        print(BOLD + OKBLUE + str(args[0]), args[1:], ENDC)
    else:
        print(BOLD + OKBLUE + str(args[0]), ENDC)
