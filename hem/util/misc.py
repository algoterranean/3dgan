"""Useful utility functions."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tqdm import tqdm
import argparse
import os

import hem



# from data import TFRecordsDataset



# TODO: improve this
def collection_to_dict(collection):
    d = {}
    for c in collection:
        name = c.name.split('/')[-1]
        name = name.split(':')[0]
        d[name] = c
    return d


# usage: list(chunks(some_list, chunk_size)) ==> list of lists of that size
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def add_to_collection(name, x):
    if type(x) == type([]):
        for v in x:
            tf.add_to_collection(name, v)
    else:
        tf.add_to_collection(name, x)



# TODO integrate this into downloading of datasets
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize



def update_moving_average(status, moving_avg, pbar):
    # calculate cumulative moving average
    if moving_avg is None:
        moving_avg = status
    else:
        for r in moving_avg:
            moving_avg[r] = (float(status[r]) + i * float(moving_avg[r])) / (i + 1)
    pbar.set_postfix(hem.format_for_terminal(moving_avg))


class CustomArgumentParser(argparse.ArgumentParser):
    """Adds support for parsing arguments from a config file."""
    def convert_arg_line_to_args(self, arg_line):
        # ignore comment and blank lines
        arg_line = arg_line.strip()
        if (len(arg_line) > 0 and arg_line[0] == '#') or len(arg_line) == 0:
            return []
        # prepend with --
        kv_pair = arg_line.split()
        k = '--' + kv_pair[0]
        return [k] + kv_pair[1:]


def inference(sess, x, summary_op, iterations, handle, handle_val, mean_image_placeholder, mean_img, desc, summary_writer, global_step):
    # sess = tf.get_default_session()
    running_total = None
    prog_bar = tqdm(range(iterations), desc=desc, unit='batch')
    for i in prog_bar:
        status = sess.run(x, feed_dict={handle: handle_val, mean_image_placeholder: mean_img })
        hem.update_moving_average(status, running_total, prog_bar)
    summary_writer.add_summary(sess.run(summary_op, feed_dict={handle: handle_val, mean_image_placeholder: mean_img }),
                               global_step=sess.run(global_step))


def init_working_dir(args):
    # save options to disk for later reference
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    f = open(os.path.join(args.dir, 'options.config'), 'w')
    for a in vars(args):
        v = getattr(args, a)
        f.write('{} {}\n'.format(a, v))
        # print('    {} = {}'.format(a, v))
    f.close()

def init_globals(args):
    # initialize globals
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
        increment_global_epoch = tf.assign(global_epoch, global_epoch+1)
        return global_step, global_epoch, increment_global_epoch

def init_summaries(dir):
    # need at least one summary present
    if len(tf.get_collection(tf.GraphKeys.SUMMARIES)) == 0:
        tf.summary.scalar('dummy_summary', tf.constant(1, tf.float32))
    # gather up all the summaries
    summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # keep summary events/writers for each phase in their own subdir
    summary_train_writer = tf.summary.FileWriter(os.path.join(dir, "train"), tf.get_default_graph())
    summary_validate_writer = tf.summary.FileWriter(os.path.join(dir, "validate"))  # , tf.get_default_graph())
    summary_test_writer = tf.summary.FileWriter(os.path.join(dir, "test"))  # , tf.get_default_graph())
    return summary_op, {'train': summary_train_writer, 'validate': summary_validate_writer, 'test': summary_test_writer}


# class HemSupervisor(tf.train.Supervisor):
class HemSupervisor(object):
    def __init__(self, args):
        with tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
            self.increment_global_epoch = tf.assign(self.global_epoch, self.global_epoch + 1)
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())
            self.summary_op, self.summary_writers = hem.init_summaries(args.dir)
            self.reset_global_step = tf.assign(self.global_step, 0)
            self.reset_global_epoch = tf.assign(self.global_epoch, 0)

        self.sv = tf.train.Supervisor(logdir=args.dir,
                                      init_op=self.init_op,
                                      summary_op=None,
                                      summary_writer=None,
                                      global_step=self.global_step,
                                      save_model_secs=0,
                                      saver=tf.train.Saver(max_to_keep=args.max_to_keep, name='saver'))



def init_supervisor(args):
    return HemSupervisor(args)



# def init_supervisor(global_step, args):
#
#     sv = tf.train.Supervisor(logdir=args.dir,
#                              init_op=init_op,
#                              summary_op=None,
#                              summary_writer=None,
#                              global_step=global_step,
#                              save_model_secs=0,
#                              saver=tf.train.Saver(max_to_keep=args.max_to_keep, name='saver'))
#     return sv
#


# def init():
#     args = hem.parse_args()
#     hem.init_working_dir(args)
#     return args