import tensorflow as tf
import numpy as np
import cv2
import matplotlib as mpl
import argparse
import math
import os
import glob
import hem

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks

# 561 x 427 ~= 211,727 patches?
# 496 x 362, stride... 12?

# some scenes from validate set:
#
# living_room_0070/scene_691
# classroom_0018/scene_271
# bedroom_0017/scene_161
# home_office_0007/scene_101
# dining_room_0026b/scene_981
# study_room_0005a/scene_21
# bedroom_0072/scene_131
# conference_room_0001/scene_341


def reset_session(sess, saver, args):
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
    return x, y, g, y_hat, y_bar #, x_full, y_full

def cgan_mean_nodes(tower=0):
    x = graph.as_graph_element('tower_{}/input_preprocess/Reshape'.format(tower)).outputs[0]
    y = graph.as_graph_element('tower_{}/input_preprocess/Reshape_1'.format(tower)).outputs[0]
    y_bar = graph.as_graph_element('tower_{}/input_preprocess/Mean'.format(tower)).outputs[0]
    # g_0 = graph.as_graph_element('tower_{}/generator/zeros_like'.format(tower)).outputs[0]
    g = graph.as_graph_element('tower_{}/generator/decoder/transpose_1'.format(tower)).outputs[0]
    y_hat = graph.as_graph_element('tower_{}/generator/add'.format(tower)).outputs[0]
    # y_0 = graph.as_graph_element('tower_{}/generator/add_1'.format(tower)).outputs[0]
    return x, y, g, y_hat, y_bar #, x_full, y_full

def read_originals(path):
    image_path = path + '_i.png'
    depth_path = path + '_f.png'
    i = cv2.imread(image_path, cv2.IMREAD_COLOR)
    i = i / np.iinfo(np.uint8).max
    i = i.astype(np.float32)
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    d = d / np.iinfo(np.uint16).max
    d = np.expand_dims(d, axis=-1)
    d = d.astype(np.float32)
    return i, d

def write_to_disk(image, depth, name, args, save_images=True):
    # write the originals back out
    original_image = image * 255.0
    original_depth = depth * 255.0
    original_depth = original_depth.astype(np.uint8)
    original_depth = cv2.applyColorMap(original_depth, cv2.COLORMAP_JET)
    if save_images:
        cv2.imwrite(os.path.join(args.dir, 'images', name + '_original_image.png'), original_image)
        cv2.imwrite(os.path.join(args.dir, 'images', name + '_original_depth.png'), original_depth)
        print('wrote {}'.format(os.path.join(args.dir, 'images', name + '_original_image.png')))
    return original_image, original_depth

def build_batch(image, x_stride=10, y_stride=10, channels=3):
    y_lim = image.shape[0]
    x_lim = image.shape[1]
    cols = int((y_lim - 65 - 29 + 1) / y_stride)
    rows = int((x_lim - 65 - 29 + 1) / x_stride)
    n = math.ceil((cols * rows) / 1024)
    image_batch = np.zeros((n * 1024, channels, 65, 65))
    c = 0
    x_pos = 0
    y_pos = 0
    for n in range(rows):
        for m in range(cols):
            patch = image[x_pos:x_pos+65, y_pos:y_pos+65, :]
            patch = np.transpose(patch, (2, 0, 1))
            image_batch[c] = patch
            c += 1
            x_pos += x_stride
        x_pos = 0
        y_pos += y_stride
    return image_batch


def forward_inference(node, image_batch, depth_batch):
    n = image_batch.shape[0]
    image_splits = np.split(image_batch, n/512)
    depth_splits = np.split(depth_batch, n/512)
    total_results = []
    for i in range(len(image_splits)):
        results = sess.run(node, feed_dict = {handle_placeholder: train_handle,
                                              x_ph: image_splits[i],
                                              y_ph: depth_splits[i]})
        total_results.append(results)

    return np.concatenate(total_results, axis=0)


def reconstruct(image_batch, depth_batch, x_stride=10, y_stride=10):
    c = 0
    x_pos = 0
    y_pos = 0
    reconstructed_image = np.zeros((427, 561, 3))
    reconstructed_depth = np.zeros((427, 561, 1))
    reconstructed_depth[:] = np.NAN
    cols = int((427 - 65 - 29 + 1) / y_stride)
    rows = int((561 - 65 - 29 + 1) / x_stride)

    for n in range(rows):
        for m in range(cols):
            # image
            i = image_batch[c]
            i = np.transpose(image_batch[c], (1, 2, 0))
            reconstructed_image[x_pos:x_pos+65, y_pos:y_pos+65, :] = i
            # depth
            d = np.transpose(depth_batch[c], (1, 2, 0))
            current_depth = reconstructed_depth[x_pos+18:x_pos+18+29, y_pos+18:y_pos+18+29, :]
            # replace NaN values with new depth patch
            new_depth = np.where(np.isnan(current_depth), d, current_depth)
            # and average
            new_depth = (new_depth + d) / 2.0
            reconstructed_depth[x_pos+18:x_pos+18+29, y_pos+18:y_pos+18+29, :] = new_depth
            c += 1
            x_pos += x_stride
        x_pos = 0
        y_pos += y_stride
    reconstructed_depth = np.nan_to_num(reconstructed_depth)
    return reconstructed_image, reconstructed_depth

def rmse(d1, d2):
    x_lim = d2.shape[0] - (18+28)
    y_lim = d2.shape[1] - (18+28)
    d1 = d1 * 10.0
    d1 = d1[18:x_lim, 18:y_lim, :]
    d2 = d2[18:x_lim, 18:y_lim, :]
    return np.sqrt(np.mean(np.square(d1 - d2)))

def process_example(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=True):
    path = '/mnt/research/datasets/nyuv2/preprocessed/' + scene + '/' + frame
    # path = os.path.join('/mnt/research/datasets/nyuv2/preprocessed/', scene, frame)
    # read in originals
    i, d = read_originals(path)
    # i, d = read_originals('/mnt/research/datasets/nyuv2/preprocessed/kitchen_0025/scene_281')
    name = scene + "_" + frame
    original_image, original_depth = write_to_disk(i, d, name, args, save_images)

    # build up the batches to feed in
    hem.message('building patches...')
    image_batch = build_batch(i, x_stride=x_stride, y_stride=y_stride)
    depth_batch = build_batch(d, x_stride=x_stride, y_stride=y_stride, channels=1)

    hem.message('generating results...')
    g_results = forward_inference(g, image_batch, depth_batch)
    y_hat_results = forward_inference(y_hat, image_batch, depth_batch)
    # g_results, y_hat_results = forward_inference(g, y_hat, image_batch, depth_batch)
    reconstructed_image_g, reconstructed_depth_g = reconstruct(image_batch, g_results, x_stride=x_stride, y_stride=y_stride)
    reconstructed_image_y_hat, reconstructed_depth_y_hat = reconstruct(image_batch, y_hat_results, x_stride=x_stride, y_stride=y_stride)

    # reconstructed image
    reconstructed_image = reconstructed_image_g * 255.0
    if save_images:
        cv2.imwrite(os.path.join(args.dir, 'images', name + '_reconstructed_image.png'), reconstructed_image)
    # variance map
    reconstructed_var = (reconstructed_depth_g - reconstructed_depth_g.min()) / (reconstructed_depth_g.max() - reconstructed_depth_g.min()) * 10.0
    reconstructed_var = reconstructed_var / 10.0 * 255.0
    reconstructed_var = reconstructed_var.astype(np.uint8)
    # reconstructed_var = cv2.applyColorMap(reconstructed_var, cv2.COLORMAP_BONE)
    if save_images:
        cv2.imwrite(os.path.join(args.dir, 'images', name + '_reconstructed_variance.png'), reconstructed_var)
    # depth map
    reconstructed_depth = reconstructed_depth_y_hat / 10.0 * 255.0
    reconstructed_depth = reconstructed_depth.astype(np.uint8)
    reconstructed_depth = cv2.applyColorMap(reconstructed_depth, cv2.COLORMAP_JET)
    if save_images:
        cv2.imwrite(os.path.join(args.dir, 'images', name + '_reconstructed_depth.png'), reconstructed_depth)
    reconstructed_var = np.concatenate((reconstructed_var, reconstructed_var, reconstructed_var), axis=2)
    montage = np.concatenate((original_image, original_depth, reconstructed_depth, reconstructed_var), axis=1)
    if save_images:
        cv2.imwrite(os.path.join(args.dir, 'images', name + '_montage.png'), montage)
    # calculate rmse
    print('\trmse for {}/{}:'.format(scene, frame), rmse(d, reconstructed_depth_y_hat))

    return montage


# def process_sample(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=True):
#     path = '/mnt/research/datasets/nyuv2/preprocessed/' + scene + '/' + frame
#     # path = os.path.join('/mnt/research/datasets/nyuv2/preprocessed/', scene, frame)
#     # read in originals
#     i, d = read_originals(path)
#     # i, d = read_originals('/mnt/research/datasets/nyuv2/preprocessed/kitchen_0025/scene_281')
#     name = scene + "_" + frame
#     original_image, original_depth = write_to_disk(i, d, name, args, save_images)
#
#     # build up the batches to feed in
#     hem.message('building patches...')
#     image_batch = build_batch(i, x_stride=x_stride, y_stride=y_stride)
#     depth_batch = build_batch(d, x_stride=x_stride, y_stride=y_stride, channels=1)
#
#     hem.message('generating results...')
#     g_results = forward_inference(g, image_batch, depth_batch)
#     y_hat_results = forward_inference(y_hat, image_batch, depth_batch)
#     # g_results, y_hat_results = forward_inference(g, y_hat, image_batch, depth_batch)
#     reconstructed_image_g, reconstructed_depth_g = reconstruct(image_batch, g_results, x_stride=x_stride,
#                                                                y_stride=y_stride)
#     reconstructed_image_y_hat, reconstructed_depth_y_hat = reconstruct(image_batch, y_hat_results, x_stride=x_stride,
#                                                                        y_stride=y_stride)
#
#     # reconstructed image
#     reconstructed_image = reconstructed_image_g * 255.0
#     if save_images:
#         cv2.imwrite(os.path.join(args.dir, 'images', name + '_reconstructed_image.png'), reconstructed_image)
#     # variance map
#     reconstructed_var = (reconstructed_depth_g - reconstructed_depth_g.min()) / (
#     reconstructed_depth_g.max() - reconstructed_depth_g.min()) * 10.0
#     reconstructed_var = reconstructed_var / 10.0 * 255.0
#     reconstructed_var = reconstructed_var.astype(np.uint8)
#     # reconstructed_var = cv2.applyColorMap(reconstructed_var, cv2.COLORMAP_BONE)
#     if save_images:
#         cv2.imwrite(os.path.join(args.dir, 'images', name + '_reconstructed_variance.png'), reconstructed_var)
#     # depth map
#     reconstructed_depth = reconstructed_depth_y_hat / 10.0 * 255.0
#     reconstructed_depth = reconstructed_depth.astype(np.uint8)
#     reconstructed_depth = cv2.applyColorMap(reconstructed_depth, cv2.COLORMAP_JET)
#     if save_images:
#         cv2.imwrite(os.path.join(args.dir, 'images', name + '_reconstructed_depth.png'), reconstructed_depth)
#     reconstructed_var = np.concatenate((reconstructed_var, reconstructed_var, reconstructed_var), axis=2)
#     montage = np.concatenate((original_image, original_depth, reconstructed_depth, reconstructed_var), axis=1)
#     if save_images:
#         cv2.imwrite(os.path.join(args.dir, 'images', name + '_montage.png'), montage)
#     # calculate rmse
#     print('\trmse for {}/{}:'.format(scene, frame), rmse(d, reconstructed_depth_y_hat))
#
#     return montage



if __name__ == '__main__':

    hem.message('Parsing arguments...')
    args = hem.parse_args()

    hem.message('Loading metafile and graph data...')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    x_ph = tf.placeholder(tf.float32, (512, 3, 65, 65))
    y_ph = tf.placeholder(tf.float32, (512, 1, 65, 65))
    # load graph, but replace input tensors with placeholders for feeding
    checkpoint_num = 50
    saver = tf.train.import_meta_graph(os.path.join(args.dir, 'checkpoint-{}.meta'.format(checkpoint_num)),
                                       input_map={"tower_0/input_preprocess/tower_0_x:0": x_ph,
                                                  "tower_0/input_preprocess/tower_0_y:0": y_ph})
    graph = tf.get_default_graph()

    hem.message('Loading dataset...')
    x, handle, iterators = hem.get_dataset_tensors(args)

    sess.run(iterators['train']['x'].initializer)
    sess.run(iterators['validate']['x'].initializer)
    train_handle = sess.run(iterators['train']['handle'])
    validate_handle = sess.run(iterators['validate']['handle'])
    handle_placeholder = graph.as_graph_element('input_pipeline/Placeholder').outputs[0]
    handle_placeholder2 = graph.as_graph_element('input_pipeline/Placeholder_1').outputs[0]

    g = graph.get_tensor_by_name("tower_0/generator/tower_0_g:0")
    y_hat = graph.get_tensor_by_name("tower_0/generator/tower_0_y_hat:0")



    reset_session(sess, saver, args)

    # generate example montages for standard model
    for i in (10, 8, 6, 4, 2, 1):
        hem.message('generating results for stride {}...'.format(i))
        ml = []
        for fn in ['bathroom_0006/scene_221',
                   'living_room_0077/scene_351',
                   'playroom_0002/scene_941',
                   'living_room_0022/scene_911',
                   'living_room_0015a/scene_661',
                   'kitchen_0003/scene_211',
                   'bookstore_0001f/scene_1631',
                   'living_room_0039/scene_641']:
            scene, frame = fn.split('/')
            m = process_example(scene, frame, g, y_hat, args, x_stride=i, y_stride=i, save_images=False)
            ml.append(m)
        full_montage = np.concatenate(ml, axis=0)
        cv2.imwrite(os.path.join(args.dir, 'images', 'full_montage_{}.png'.format(i)), full_montage)

    # generate example montages for sampler model




    #
    # hem.message('generating results for sampler...')
    # scene = 'bathroom_0006'
    # frame = 'scene_221'
    # m1 = process_example(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=False)
    # m2 = process_example(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=False)
    # m3 = process_example(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=False)
    # m4 = process_example(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=False)
    # m5 = process_example(scene, frame, g, y_hat, args, x_stride=10, y_stride=10, save_images=False)
    # full_montage = np.concatenate([m1, m2, m3, m4, m5], axis=0)
    # cv2.imwrite(os.path.join(args.dir, 'images', 'full_montage_sampler.png'), full_montage)
    #

    # # TO SELECT IMAGES AT RANDOM FROM VALIDATION SET:
    # fn = '/mnt/research/datasets/nyuv2/preprocessed/validation.txt'
    # f = open(fn, 'r')
    # lines = [line.strip() for line in f]
    # import random
    # for i in range(8):
    #     print(random.choice(lines))

