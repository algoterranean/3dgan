import tensorflow as tf
import time
import re
from sys import stdout
from util import print_progress, fold


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







            
