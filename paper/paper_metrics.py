import tensorflow as tf
import numpy as np
import cv2
import os
from collections import Counter
import hem

# TODO
# - add support for nan results in metrics (ignore and adjust mean accordingly?)


def metrics(x, y, g, y_hat):
    g = g / 10.0
    y = y / 10.0
    y_hat = y_hat / 10.0
    linear_rmse = hem.rmse(y, y_hat)
    log_rmse = hem.rmse(tf.log(y + 1e-8), tf.log(y_hat + 1e-8))
    abs_rel_diff = tf.reduce_mean(tf.abs(y - y_hat) / y_hat)
    squared_rel_diff = tf.reduce_mean(tf.square(y - y_hat) / y_hat)
    d = tf.log(y + 1e-8) - tf.log(y_hat + 1e-8)
    n = tf.cast(tf.size(d), tf.float32)  # tf.size() = 430592
    scale_invariant_log_rmse = tf.reduce_mean(tf.square(d)) - (tf.reduce_sum(d) ** 2) / (n ** 2)
    delta = tf.maximum(y / y_hat, y_hat / y)
    t1, t1_op = tf.metrics.percentage_below(delta, 1.25, name='threshold1')
    t2, t2_op = tf.metrics.percentage_below(delta, 1.25 ** 2, name='threshold2')
    t3, t3_op = tf.metrics.percentage_below(delta, 1.25 ** 3, name='threshold3')
    all_metrics = {'linear_rmse': linear_rmse,
                   'log_rmse': log_rmse,
                   'abs_rel_diff': abs_rel_diff,
                   'squared_rel_diff': squared_rel_diff,
                   'scale_invariant_log_rmse': scale_invariant_log_rmse,
                   't1': t1_op,
                   't2': t2_op,
                   't3': t3_op}
    return all_metrics

def reset_session(args):
    hem.message('Resetting variables...')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    hem.message('Restoring checkpoint...')
    latest = tf.train.latest_checkpoint(args.dir)
    print(latest)
    saver.restore(sess, latest)


def colorize_depthmap(depth):
    depth = np.transpose(depth, [1, 2, 0])
    depth = depth * np.iinfo(np.uint8).max
    depth = depth.astype(np.uint8)
    colorized = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth, colorized

def cgan_baseline_nodes(tower=0):
    x = graph.as_graph_element('tower_{}/input_preprocess/Reshape'.format(tower)).outputs[0]
    y = graph.as_graph_element('tower_{}/input_preprocess/Reshape_1'.format(tower)).outputs[0]
    y_bar = graph.as_graph_element('tower_{}/input_preprocess/Mean'.format(tower)).outputs[0]
    # g_0 = graph.as_graph_element('tower_{}/generator/zeros_like'.format(tower)).outputs[0]
    g = graph.as_graph_element('tower_{}/generator/decoder/transpose_1'.format(tower)).outputs[0]
    y_hat = g
    # y_0 = g_0
    return x, y, g, y_hat, y_bar

def cgan_mean_nodes(tower=0):
    x = graph.as_graph_element('tower_{}/input_preprocess/Reshape'.format(tower)).outputs[0]
    y = graph.as_graph_element('tower_{}/input_preprocess/Reshape_1'.format(tower)).outputs[0]
    y_bar = graph.as_graph_element('tower_{}/input_preprocess/Mean'.format(tower)).outputs[0]
    # g_0 = graph.as_graph_element('tower_{}/generator/zeros_like'.format(tower)).outputs[0]
    g = graph.as_graph_element('tower_{}/generator/decoder/transpose_1'.format(tower)).outputs[0]
    y_hat = graph.as_graph_element('tower_{}/generator/add'.format(tower)).outputs[0]
    # y_0 = graph.as_graph_element('tower_{}/generator/add_1'.format(tower)).outputs[0]
    return x, y, g, y_hat, y_bar


hem.message('Parsing arguments...')
args = hem.parse_args()

hem.message('Loading metafile and graph data...')
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.import_meta_graph(os.path.join(args.dir, 'checkpoint-50.meta'))
graph = tf.get_default_graph()

hem.message('Loading dataset...')
x, handle, iterators = hem.get_dataset_tensors(args)
sess.run(iterators['train']['x'].initializer)
sess.run(iterators['validate']['x'].initializer)
train_handle = sess.run(iterators['train']['handle'])
validate_handle = sess.run(iterators['validate']['handle'])
handle_placeholder = graph.as_graph_element('input_pipeline/Placeholder').outputs[0]
# mean_image_placeholder = graph.as_graph_element('Placeholder').outputs[0]

# get needed Tensors from model
if args.model_version == 'baseline':
    x, y, g, y_hat, y_bar = cgan_baseline_nodes(0)
    x2, y2, g2,  y_hat2, y_bar2 = cgan_baseline_nodes(1)
else:
    x, y, g, y_hat, y_bar = cgan_mean_nodes(0)
    x2, y2, g2, y_hat2, y_bar2 = cgan_mean_nodes(1)

m1 = metrics(x, y, g, y_hat)
m2 = metrics(x2, y2, g2, y_hat2)
mean_image_placeholder = tf.placeholder(tf.float32, (args.batch_size, 1, 29, 29))
m_mean_1 = metrics(x, y, mean_image_placeholder, mean_image_placeholder)
m_mean_2 = metrics(x2, y2, mean_image_placeholder, mean_image_placeholder)
if args.model_version == 'baseline':
    m_g0_1 = metrics(x, y, mean_image_placeholder, mean_image_placeholder)
    m_g0_2 = metrics(x2, y2, mean_image_placeholder, mean_image_placeholder)
else:
    m_g0_1 = metrics(x, y, mean_image_placeholder, mean_image_placeholder + y_bar)
    m_g0_2 = metrics(x2, y2, mean_image_placeholder, mean_image_placeholder + y_bar2)
zero_image_batch = np.zeros((args.batch_size, 1, 29, 29))



def calculate_metrics(dataset_name, dataset_handle, n_batches):
    reset_session(args)
    hem.message('Calculating metrics for {} set...'.format(dataset_name))
    # accumulate results
    results = sess.run([m1, m2, y, y2], feed_dict={handle_placeholder: dataset_handle})
    g_metrics = Counter(results[0]) + Counter(results[1])
    mean_image = np.concatenate((results[2], results[3]), axis=0)
    for i in range(n_batches - 1):
        results = sess.run([m1, m2, y, y2], feed_dict={handle_placeholder: dataset_handle})
        g_metrics = g_metrics + Counter(results[0]) + Counter(results[1])
        mean_image = np.concatenate((mean_image, results[2], results[3]), axis=0)
    # average metrics
    n = n_batches * 2
    hem.message('Model metrics:')
    for k in ['t1', 't2', 't3', 'abs_rel_diff', 'squared_rel_diff', 'linear_rmse', 'log_rmse', 'scale_invariant_log_rmse']:
        print('\t{}: {:.3f}'.format(k, g_metrics[k]/n))
    # process mean image
    mean_image = np.mean(mean_image, axis=0)
    mean_depth, mean_depth_colorized = colorize_depthmap(mean_image)
    # print('IMAGE:', os.path.join(args.dir, 'metrics', 'test_mean.png'))
    cv2.imwrite(os.path.join(args.dir, 'metrics', '{}_mean.png'.format(dataset_name)), mean_depth)
    cv2.imwrite(os.path.join(args.dir, 'metrics', '{}_mean_colorized.png'.format(dataset_name)), mean_depth_colorized)

    # calculate metrics using mean dataset depth
    mean_image_batch = np.stack([mean_image]*args.batch_size, axis=0)
    results = sess.run([m_mean_1, m_mean_2], feed_dict={handle_placeholder: dataset_handle,
                                                        mean_image_placeholder: mean_image_batch})
    mean_metrics = Counter(results[0]) + Counter(results[1])
    for i in range(n_batches - 1):
        results = sess.run([m_mean_1, m_mean_2], feed_dict={handle_placeholder: dataset_handle,
                                                            mean_image_placeholder: mean_image_batch})
        mean_metrics = mean_metrics + Counter(results[0]) + Counter(results[1])
    hem.message('Mean metrics:')
    for k in ['t1', 't2', 't3', 'abs_rel_diff', 'squared_rel_diff', 'linear_rmse', 'log_rmse', 'scale_invariant_log_rmse']:
        print('\t{}: {:.3f}'.format(k, mean_metrics[k]/n))

    # calculate metrics using g = 0
    results = sess.run([m_g0_1, m_g0_2], feed_dict={handle_placeholder: dataset_handle,
                                                    mean_image_placeholder: zero_image_batch})
    # print(results)
    zero_metrics = Counter(results[0]) + Counter(results[1])
    for i in range(n_batches - 1):
        results = sess.run([m_g0_1, m_g0_2], feed_dict={handle_placeholder: dataset_handle,
                                                            mean_image_placeholder: zero_image_batch})
        # print(results)
        zero_metrics = zero_metrics + Counter(results[0]) + Counter(results[1])
    hem.message('Zero metrics:')
    for k in ['t1', 't2', 't3', 'abs_rel_diff', 'squared_rel_diff', 'linear_rmse', 'log_rmse', 'scale_invariant_log_rmse']:
        print('\t{}: {:.3f}'.format(k, zero_metrics[k]/n))

calculate_metrics('validate', validate_handle, iterators['validate']['batches']) # * 10)
calculate_metrics('train', train_handle, iterators['train']['batches']) # * 2)




# TODO
# - calculate y_0 for mean_provided and mean_adjusted
# - calculate y_mean for baseline

