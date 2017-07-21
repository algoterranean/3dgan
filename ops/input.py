"""Input pipeline operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_slice(x, batch_size, slice_index, name=None):
    """Returns a slice of x appropriate for this GPU (starting at gpu_id=0)
    
    Args:
      x: Tensor, the tensor to slice up.
      batch_size: Integer, batch size of resulting slice.
      slice_index: Integer, which slice to return.
      name: String, name for name scope. 

    Returns:
      A tensor representing a slice of the original input tensor.
    """
    with tf.name_scope(name, 'batch_slice', [x]):
        y = x[slice_index * batch_size:(slice_index+1) * batch_size, :]
    return y
