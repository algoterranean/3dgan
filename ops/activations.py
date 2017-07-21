"""Implementation of common activation functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def lrelu(x, leak=0.2, name=None):
    """Leakly ReLU activation function.

    Args:
      x: Tensor, input tensor.
      leak: Float, the leak value to use. Should be in range [0,1].
      name: String, name for the output tensor.

    Returns:
      A tensor representing x with LReLU applied. 

    Source:
    ------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models`
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    """
    with tf.name_scope(name, 'lelu', [x]):
        y = tf.maximum(leak*x, x, name=name)
    return y


def selu(x, alpha=1.67236, scale=1.507, name=None):
    """Scaled elU activation function.

    When using this:
      - scale inputs to zero mean and unit variance.
      - initialize weights with stddev=sqrt(1/n)
      - use selu dropout?

    Args:
      x: Tensor, input tensor.
      alpha: 
      scale:
      name:

    Source:
    ------
    - `Self-Normalizing Neural Networks`
    https://arxiv.org/abs/1706.02515
    """
    with tf.name_scope(name, 'selu', [x]):
        y = scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    return y
