import tensorflow as tf
import numpy as np
import argparse
import os
from models import simple_fc, simple_cnn


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()

sess = tf.Session()


saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))


graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()

a = graph.as_graph_element('outputs/decoder/Layer.Output')
b = graph.as_graph_element('inputs/x_input')
# # print(a)
y_hat = a.outputs[0]
x_input = b.outputs[0]

xs = np.random.random([1, 64, 64, 3])

c = sess.run([y_hat], {x_input: xs})
print(c)

# print('Graph:', graph)
# print('GraphDef:', graph_def)


# for node in graph_def.node:
#     print(node.name)
#     # print(node.op)
#     # # if node.op == 'Variable':
#     # #     print(node.name)


#     # tf.Variable('outputs/decoder/Layer.Output')





# print('Collections:')
# for c in graph.get_all_collection_keys():
#     print('Collection', c)
#     for v in tf.get_collection(c):
#         print(v)

# print('Operations:')
# for o in graph.get_operations():
#     print(o.name)


# print('Global vars:')
# for var in tf.global_variables():
#     print(var.op.name)



# x_input = graph.get_operation_by_name('inputs/x_input')
# x = graph.get_operation_by_name('inputs/x')
# print('x_input:', type(x_input))
# print('x:', type(x))


# print(x)
# print(x_input)


# all_vars = tf.get_collection('layers')
# first_layer = all_vars[0]
# print('layer:')
# print(first_layer)




# xs = np.random.random([1, 64, 64, 3])
# sess.run([first_layer], feed_dict={x_input: xs})




    

