"""Useful utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import pickle
import re

# from data import TFRecordsDataset
from ops.input import batch_slice



def checkpoints(workspace_dir, only_recent=False):
    checkpoint_files = []
    for f in os.listdir(os.path.join(workspace_dir, 'checkpoints')):
        if f.endswith('.meta'):
            checkpoint_files.append(f[:f.rfind('.')])
    checkpoint_files.sort()
    if only_recent:
        return [checkpoint_files[-1]]
    return checkpoint_files


def reload_session(path, fn=None):
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.import_meta_graph(os.path.join(path, 'model'))
    if fn is None:
        chk_file = tf.train.latest_checkpoint(os.path.join(path, 'checkpoints'))
    else:
        chk_file = fn
    saver.restore(sess, chk_file)
    return sess