"""Implementation of common activation functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def lrelu(x, leak=0.2, name=None):
    """Leakly ReLU activation function.

    Sources:
    -------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models`
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    """
    return tf.maximum(leak*x, x, name=name)
