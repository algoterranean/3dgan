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


