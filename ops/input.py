"""Input pipeline operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def input_slice(x, batch_size, gpu_id, name=None):
    """
    Returns a slice of x appropriate for this GPU (starting at gpu_id=0)
    """
    return x[gpu_id * batch_size : (gpu_id+1) * batch_size, :]
