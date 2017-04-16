import tensorflow as tf


class Model:

    def __init__(self, x):
        self.output = None
        self.summary_nodes = []
        self._build_graph(x)

    def _add_summary(self, node, name):
        self.summary_nodes.append(tf.summary.histogram(name, node))        
