import tensorflow as tf
import time
import re
from sys import stdout
from util import print_progress, fold, average_gradients


class Model:
    """Parent class for all models used in this framework.
    Each model should set self.train_op and self.summary_op in their init()."""
        
    def tensor_name(self, x):
        return re.sub('tower_[0-9]*/', '', x.op.name)

    
    def activation_summary(self, x):
        n = self.tensor_name(x)
        tf.summary.histogram(n + '/activations', x)
        tf.summary.scalar(n + '/sparsity', tf.nn.zero_fraction(x))
        
    def variables_on_cpu(self, gpu_id):
        def helper(op):
            return '/cpu:0' if op.type == 'VariableV2' else '/gpu:{}'.format(gpu_id)
        return helper

    def summarize_collection(self, name, scope):
        collection = tf.get_collection(name, scope)
        for x in collection:
            tf.summary.scalar(self.tensor_name(x), x)
        return collection

    def average_and_apply_grads(self, tower_grads, opt):
        with tf.variable_scope('average_gradients'):
            avg_grads = average_gradients(tower_grads)
        with tf.variable_scope('apply_gradients'):
            apply_grads = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())
        with tf.variable_scope('train_op'):
            train_op = tf.group(apply_grads)
        return avg_grads, apply_grads, train_op

    def summarize_grads(self, grads):
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)


