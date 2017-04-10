import tensorflow as tf


def variational_autoencoder(x):
    orig_shape = x.get_shape().as_list()
    summary_nodes = []

    
