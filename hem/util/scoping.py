"""Useful utility functions."""

from __future__ import absolute_import, division, print_function

import re

import tensorflow as tf

# from data import TFRecordsDataset
# from hem import batch_slice
# from hem.ops import batch_slice

import hem


def tensor_name(x):
    """Remove tower prefix from tensor name.

    Args:
      x: Tensor, a tensor scoped to a GPU tower.

    Returns:
      A string with the name of the tensor without the tower name.
    """
    return re.sub('tower_[0-9]*/', '', x.op.name)


def variables_on_cpu(gpu_id):
    """Keeps variables on the CPU when used in tf.device().

    This can be passed to a `tf.device()` call to dynamically decide
    where to put ops and tensors within that device scope.

    Args:
      gpu_id: Integer, GPU to use if CPU is not appropriate.

    Returns:
      Helper function that returns a GPU or CPU device id.
    """
    def helper(op):
        return '/cpu:0' if op.type == 'VariableV2' else '/gpu:{}'.format(gpu_id)
    return helper




def default_to_cpu(func):
    """Decorator to default all variables to the CPU.

    Primarily used for defining model init functions.

    Args:
      func: Function, function to call within CPU scope.

    Returns:
      A decorator/wrapper function.
    """

    def wrapper(*args, **kwargs):
        with tf.device('/cpu:0'):
            return func(*args, **kwargs)

    return wrapper

# TODO...
def tower_scope_range(x, n_gpus, batch_size):
    """Scope and sliced input generator for multi-GPU environment.

    This iterator can be used to wrap around model construction and will
    generate an appropriate GPU scope and a slice of the input tensor.
    This iterator will automatically reuse variable scope after first
    tower.

    Args:
      x: Tensor, the input tensor(s) to this model.
      n_gpus: Integer, number of GPUs available.
      batch_size: Integer, size of the slice to be created for this GPU
      from the original input tensor.

    Returns:
      An iterator that returns a 3 tuple containing the input for this
      GPU, the scope, and the GPU's id.
    """
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_id in range(n_gpus):
            with tf.device(variables_on_cpu(gpu_id)):
                with tf.name_scope('tower_{}'.format(gpu_id)) as scope:
                    yield hem.batch_slice(x, batch_size, gpu_id), scope, gpu_id
                    tf.get_variable_scope().reuse_variables()
