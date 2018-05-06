
"""Support for NYUv2 dataset.

See http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html for more info.

This dataset is quite large and requires significant pre-processing before
it can be used. The general idea is

1) Download the entire RAW dataset
2) Download the MATLAB toolbox
3) Follow the toolbox instructions to generate matching RGB/depth frame-pairs
4) ...
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import TFRecordDataset
from tqdm import tqdm
import cv2
import math
import os
import hem
import random

# TODO use tf.extract_image_patches to pull out multiple patches per image
# TODO add data augmentation

_features = {'image':          tf.FixedLenFeature([], tf.string),
             'depth':          tf.FixedLenFeature([], tf.string),
             'width':          tf.FixedLenFeature([], tf.int64),
             'height':         tf.FixedLenFeature([], tf.int64),
             'channels':       tf.FixedLenFeature([], tf.int64),
             'filename':       tf.FixedLenFeature([], tf.string),
             'depth_filename': tf.FixedLenFeature([], tf.string)}

_dataset_files = {'train':    'nyuv2.train.tfrecords',
                  'validate': 'nyuv2.validate.tfrecords',
                  'test':     'nyuv2.test.tfrecords'}

args = {
    '--resize': {
        'type': int,
        'nargs': 2,
        'help': """Resize input images to size w x h. This argument, if specified, requires 
        two values (width and height)."""
        },
    '--random_crop': {
        'type': int,
        'nargs': 2,
        'help': """Randomly crop the input images to size h x w.  This argument, if specified, 
        requires two values (width and height)."""
        },
    '--include_location': {
        'action': 'store_true',
        'default': False,
        'help': """If True and using random_crop, will include a 2-channel location information
        with the images and depths, indicating where in the image (as a percentage of the W/H) the crop
        was taken from."""
        },
    '--skip_invalid': {
        'action': 'store_true',
        'default': False,
        'help': """Ignores images in the dataset that have gaps  (values that are 0 or uint16.max, 
        indicating a failure of the Kinect sensor to get any return signature during capture)."""
        },
    '--normalize': {
        'action': 'store_true',
        'default': False,
        'help': 'Subtracts the mean (per-image) depth from each image, and provides the mean in the dataset.'
    },

    '--include_originals': {
        'type': int,
        'nargs': 2,
        'help': 'Includes the original x and y images, resized to the provided dimensions.'
    }


    # '--augment': {
    #     'action': 'store_true',
    #     'default': False,
    #     'help': 'Augment the data.'
    #     },
    # '--samples_per_image': {
    #     'default': 1,
    #     'type': int,
    #     'help': 'Number of random crops to pick per image.'
    #     }
    }


class NYUv2Dataset(hem.DataPlugin):
    name = 'nyuv2'

    @staticmethod
    def arguments():
        return args

    @staticmethod
    def check_prepared_datasets(storage_dir):
        file_list = [v for k, v in _dataset_files.items()]
        return NYUv2Dataset.check_files(storage_dir, file_list)

    @staticmethod
    def check_raw_datasets(storage_dir):
        files = os.listdir(storage_dir)
        for f in ['train.txt', 'validation.txt', 'test.txt']:
            if f not in files:
                return False
        return True

    @staticmethod
    def download(download_dir):
        return False
        # raise NotImplementedError("""No support for downloading NYUv2. Please download
        # the raw dataset and run the preprocessing code available from the website.""")

    @staticmethod
    def convert_to_tfrecord(dataset_dir, storage_dir):
        def build_dataset(name, file_list):
            writer = tf.python_io.TFRecordWriter('nyuv2.{}.tfrecords'.format(name))
            image_dir = os.path.join(dataset_dir)
            lines = open(os.path.join(image_dir, file_list)).readlines()
            for line in tqdm(lines):
                fin = os.path.join(image_dir, line.strip() + '_i.png')
                fdn = os.path.join(image_dir, line.strip() + '_f.png')
                image_data = tf.gfile.FastGFile(fin, 'rb').read()
                depth_data = tf.gfile.FastGFile(fdn, 'rb').read()
                w, h, c = 427, 561, 3
                # write record
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': hem.bytes_feature(image_data),
                        'depth': hem.bytes_feature(depth_data),
                        'width': hem.int64_feature(w),
                        'height': hem.int64_feature(h),
                        'channels': hem.int64_feature(c),
                        'filename': hem.bytes_feature(tf.compat.as_bytes(fin)),
                        'depth_filename': hem.bytes_feature(tf.compat.as_bytes(fdn))
                        }))
                writer.write(example.SerializeToString())
        build_dataset('train', 'train.txt')
        build_dataset('validate', 'validation.txt')
        build_dataset('test', 'test.txt')

    @staticmethod
    def parse_tfrecord(args):
        def helper(proto):
            parsed = tf.parse_single_example(proto, _features)
            w = tf.cast(parsed['width'], tf.int32)
            h = tf.cast(parsed['height'], tf.int32)
            c = tf.cast(parsed['channels'], tf.int32)
            image = tf.image.decode_png(parsed['image'], channels=3, dtype=tf.uint8)
            depth = tf.image.decode_png(parsed['depth'], channels=1, dtype=tf.uint16)
            # image = tf.image.decode_image(parsed['image'], channels=3)
            # depth = tf.image.decode_image(parsed['depth'], channels=1)
            x_vec = tf.lin_space(0.0, 1.0, h)
            y_vec = tf.lin_space(0.0, 1.0, w)
            x_loc_channel = tf.stack([x_vec] * 427)
            x_loc_channel = tf.expand_dims(x_loc_channel, axis=-1)
            y_loc_channel = tf.stack([y_vec] * 561, axis=1)
            y_loc_channel = tf.expand_dims(y_loc_channel, axis=-1)
            # add shape data back
            image = tf.reshape(image, tf.stack([w, h, c]))
            depth = tf.reshape(depth, tf.stack([w, h, 1]))

            if args.include_originals:
                # x_full = image
                # y_full = depth
                x_full = tf.cast(image, tf.float32)
                y_full = tf.cast(depth, tf.float32)
                x_full = tf.image.resize_images(x_full, args.include_originals)
                y_full = tf.image.resize_images(y_full, args.include_originals)
                x_full = tf.transpose(x_full, [2, 0, 1])
                y_full = tf.transpose(y_full, [2, 0, 1])
                x_full = x_full / tf.uint8.max
                y_full = y_full / tf.uint16.max
                # x_full = tf.cast(x_full, tf.float32) / 255.0
                # y_full = tf.cast(y_full, tf.float32) / 255.0

                # if args.include_location:
                #     x_loc_channel = tf.image.resize_images(x_loc_channel, args.include_originals)
                #     y_loc_channel = tf.image.resize_images(y_loc_channel, args.include_originals)

            if args.resize:
                image = tf.image.resize_images(image, args.resize)
                depth = tf.image.resize_images(depth, args.resize)
                if args.include_location:
                    x_loc_channel = tf.image.resize_images(x_loc_channel, args.resize)
                    y_loc_channel = tf.image.resize_images(y_loc_channel, args.resize)

            if args.random_crop:
                if args.include_location:
                    image = tf.cast(image, tf.float32)
                    depth = tf.cast(depth, tf.float32)
                    combined = tf.concat([image, depth, x_loc_channel, y_loc_channel], axis=-1)
                    combined = tf.random_crop(combined, args.random_crop + [6], seed=args.seed)
                    image, depth, x_loc, y_loc = tf.split(combined, num_or_size_splits=[3, 1, 1, 1], axis=-1)
                    x_loc = tf.transpose(x_loc, [2, 0, 1])
                    y_loc = tf.transpose(y_loc, [2, 0, 1])
                    image = tf.cast(image, tf.float32)
                    depth = tf.cast(depth, tf.float32)
                else:
                    image = tf.cast(image, tf.float32)
                    depth = tf.cast(depth, tf.float32)
                    combined = tf.concat([image, depth], axis=-1)
                    combined = tf.random_crop(combined, args.random_crop + [4], seed=args.seed)
                    image, depth = tf.split(combined, num_or_size_splits=[3, 1], axis=-1)

            # convert to [0, 1]
            # image = tf.cast(image, tf.float32) / 255.0
            # depth = tf.cast(depth, tf.float32) / 255.0
            image = image / tf.uint8.max
            depth = depth / tf.uint16.max

            # image = tf.cast(image, tf.float32) / tf.uint8.max
            # depth = tf.cast(depth, tf.float32) / tf.uint16.max

            # normalize
            if args.normalize:
                mean = tf.reduce_mean(depth)
                mean_vec = tf.ones_like(depth) * mean
                mean_vec = tf.transpose(mean_vec, [2, 0, 1])

                # depth = depth - mean
                # depth = (depth - tf.reduce_min(depth)) / (tf.reduce_max(depth) - tf.reduce_min(depth))

                # # standardize to gaussian normal
                # depth = tf.image.per_image_standardization(depth)
                # # standardize back to uniform
                # depth = 0.5 * tf.erfc(- depth / 1.4142135624)

                # depth = tf.sigmoid(depth)
                # image = tf.image.per_image_standardization(image)
                # depth = depth - mean
                # depth = (depth - tf.reduce_min(depth)) / (tf.reduce_max(depth) - tf.reduce_min(depth))

            # convert to NCHW format
            image = tf.transpose(image, [2, 0, 1])
            depth = tf.transpose(depth, [2, 0, 1])
            # if args.inverse_depth:
            #     depth = tf.reciprocal(depth)

            output = [image, depth]
            if args.random_crop and args.include_location:
                output += [x_loc, y_loc]
            if args.normalize:
                output += [mean_vec] #tf.expand_dims(mean, 0)]
            if args.include_originals:
                output += [x_full, y_full]
            return output

        return helper


    @staticmethod
    def get_datasets(args):
        def ignore_incomplete_depthmaps(x, y, *args):
            return tf.logical_not(tf.reduce_any(tf.logical_or(tf.equal(y, tf.ones_like(y)),
                                                              tf.equal(y, tf.zeros_like(y)))))
        datasets = {}
        for k, v in _dataset_files.items():
            fn = os.path.join(args.dataset_dir, v)
            dataset = TFRecordDataset(fn)
            dataset = dataset.map(NYUv2Dataset.parse_tfrecord(args), num_threads=args.n_threads)
            dataset = dataset.filter(ignore_incomplete_depthmaps)
            datasets[k] = (dataset, fn)
        return datasets


if __name__ == '__main__':
    import argparse
    import hem
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dataset_dir', default='/mnt/research/datasets/nyuv2/preprocessed')
    p.add_argument('--dataset_dir', default='/mnt/research/projects/autoencoders/data/storage')
    p.add_argument('--dataset', default='nyuv2')
    args = p.parse_args()
    print('Attempting to get datasets...')
    datasets = hem.get_datasets(args)
    print('Finished! Received datasets', dataset)
