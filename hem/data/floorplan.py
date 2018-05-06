import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from tqdm import tqdm
import hem

# from hem.data.DataPlugin import DataPlugin, _bytes_feature, _int64_feature

# TODO what to do about "Premature end of JPEG file?"
# Can we catch it? or... ?


_features = {
    'image': tf.FixedLenFeature([], tf.string),
    'width': tf.FixedLenFeature([], tf.int64),
    'height': tf.FixedLenFeature([], tf.int64),
    'channels': tf.FixedLenFeature([], tf.int64),
    'filename': tf.FixedLenFeature([], tf.string)
    }

_output_files = ['floorplan.train.tfrecords',
                 'floorplan.validate.tfrecords',
                 'floorplan.test.tfrecords']


class FloorplanDataset(hem.DataPlugin):
    name = 'floorplan'

    # TODO should take an args struct for the params
    @staticmethod
    def check_prepared_datasets(storage_dir):
        for fn in _output_files:
            if not os.path.exists(os.path.join(storage_dir, fn)):
                return False
        return True
        # return FloorplanDataset.check_files(storage_dir, _output_files)

    # TODO should take an args struct for the params
    @staticmethod
    def check_raw_datasets(download_dir):
        # check that the directory contains the necessary lists of files
        for fn in ['train_set.txt', 'validation_set.txt', 'test_set.txt']:
            if not os.path.exists(os.path.join(download_dir, fn)):
                return False
        return True
        # files = os.listdir(storage_dir)
        # for f in ['train_set.txt', 'validation_set.txt', 'test_set.txt']:
        #     if f not in files:
        #         return False
        # return True

    @staticmethod
    def download(download_dir):
        pass
        # raise NotImplementedError("There is currently no support for downloading the floorplan dataset.")

    # TODO should take an args struct for the params
    @staticmethod
    def convert_to_tfrecord(dataset_dir, storage_dir):
        def build_dataset(name, file_list):
            writer = tf.python_io.TFRecordWriter(os.path.join(storage_dir, 'floorplans.{}.tfrecords'.format(name)))
            image_dir = os.path.join(dataset_dir)  # '/mnt/research/datasets/floorplans')
            lines = open(os.path.join(image_dir, file_list)).readlines()
            for line in tqdm(lines):
                fn = os.path.join(image_dir, line.strip())
                # read image
                with tf.gfile.FastGFile(fn, 'rb') as f:
                    image_data = f.read()
                # determine shape
                f = open(fn, 'rb')
                stuff = f.read()
                f.close()
                nparr = np.fromstring(stuff, np.uint8)
                i = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                w, h, c = i.shape
                # write record
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': hem.bytes_feature(image_data),
                        'width': hem.int64_feature(w),
                        'height': hem.int64_feature(h),
                        'channels': hem.int64_feature(c),
                        'filename': hem.bytes_feature(tf.compat.as_bytes(fn))}))
                writer.write(example.SerializeToString())

        build_dataset('train', 'train_set.txt')
        build_dataset('validate', 'validation_set.txt')
        build_dataset('test', 'test_set.txt')

    @staticmethod
    def get_datasets(args):
        train_fn = os.path.join(args.dataset_dir, _output_files[0])
        train_set = TFRecordDataset(train_fn)
        validate_fn = os.path.join(args.dataset_dir, _output_files[1])
        validate_set = TFRecordDataset(validate_fn)
        test_fn = os.path.join(args.dataset_dir, _output_files[2])
        test_set = TFRecordDataset(test_fn)

        train_set = train_set.map(FloorplanDataset.parse_tfrecord(args))
        validate_set = validate_set.map(FloorplanDataset.parse_tfrecord(args))
        test_set = test_set.map(FloorplanDataset.parse_tfrecord(args))
        return {'train': (train_set, train_fn), 'validate': (validate_set, validate_fn), 'test': (test_set, test_fn)}

    @staticmethod
    def parse_tfrecord(args):
        def helper(example_proto):
            parsed = tf.parse_single_example(example_proto, _features)
            image = tf.image.decode_image(parsed['image'], channels=3)
            w = tf.cast(parsed['width'], tf.int32)
            h = tf.cast(parsed['height'], tf.int32)
            c = tf.cast(parsed['channels'], tf.int32)
            image_shape = tf.stack([w, h, c])
            image = tf.reshape(image, image_shape)
            image = tf.image.resize_images(image, [64, 64])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.reshape(image, (64, 64, 3))
            image = tf.transpose(image, [2, 0, 1])
            return image
        return helper


if __name__ == '__main__':
    # generate_dataset('train', 'train_set.txt')
    # generate_dataset('test', 'test_set.txt')
    # generate_dataset('validate', 'validation_set.txt')    
    fn = 'floorplans.test.tfrecords'
    c = sum([1 for r in tf.python_io.tf_record_iterator(fn)])
    print('floorplans test:', c)
