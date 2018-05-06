import gzip
import os
import struct
import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
import hem

# from hem.data.DataPlugin import DataPlugin, _bytes_feature, _int64_feature

_input_files = ['t10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz',
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz']

_output_files = ['mnist.train.tfrecords',
                 'mnist.test.tfrecords']

_features = {
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
    }


class MNISTDataset(hem.DataPlugin):
    name = 'mnist'

    @staticmethod
    def check_raw_datasets(storage_dir):
        return MNISTDataset.check_files(storage_dir, _input_files)

    @staticmethod
    def check_prepared_datasets(storage_dir):
        return MNISTDataset.check_files(storage_dir, _output_files)

    @staticmethod
    def download(download_dir):
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        try:
            for f in _input_files:
                urllib.request.urlretrieve(base_url + f, os.path.join(download_dir, f))
        except Exception as e:
            print(e)
            return False
        return True

    @staticmethod
    def convert_to_tfrecord(download_dir, storage_dir):
        def read_images(fn):
            with gzip.open(os.path.join(download_dir, fn)) as f:
                data = f.read()
            magic_num, num_images, num_rows, num_cols = struct.unpack('>iiii', data[0:4*4])
            images = np.frombuffer(data[4*4:], dtype=np.uint8)
            images = np.reshape(images, (num_images, num_rows, num_cols, 1))
            return images

        def read_labels(fn):
            with gzip.open(os.path.join(download_dir, fn)) as f:
                data = f.read()
            magic_num, num_labels = struct.unpack('>ii', data[0:4*2])
            labels = np.frombuffer(data[2*4:], dtype=np.uint8)
            labels = np.reshape(labels, (num_labels, 1))
            return labels

        def write_tfrecord(images, labels, fn):
            writer = tf.python_io.TFRecordWriter(os.path.join(storage_dir, fn))
            for i in range(images.shape[0]):
                img = images[i].tostring()
                label = int(labels[i])
                record = tf.train.Example(features=tf.train.Features(feature={
                    'image': hem.bytes_feature(img),
                    'label': hem.int64_feature(label)
                    }))
                writer.write(record.SerializeToString())
            writer.close()

        write_tfrecord(read_images(_input_files[0]),
                       read_labels(_input_files[1]),
                       'mnist.test.tfrecords')
        write_tfrecord(read_images(_input_files[2]),
                       read_labels(_input_files[3]),
                       'mnist.train.tfrecords')

    @staticmethod
    def get_datasets(args):
        test_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[0]))
        train_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[1]))
        test_set = test_set.map(MNISTDataset.parse_tfrecord)
        train_set = train_set.map(MNISTDataset.parse_tfrecord)
        return {'train': train_set, 'validate': None, 'test': test_set}

    @staticmethod
    def parse_tfrecord(args):
        def helper(example_proto):
            parsed = tf.parse_single_example(example_proto, _features)
            # decode image, shape it, and convert to [0, 1]
            image = tf.decode_raw(parsed['image'], tf.uint8)
            image = tf.reshape(image, [28, 28, 1])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.transpose(image, [2, 0, 1])
            # label
            label = parsed['label']
            return image, label
        return helper


if __name__ == '__main__':
    pass

    # # all optional kwargs
    # from hem.util.misc import TqdmUpTo
    # with TqdmUpTo(unit='B',
    #               unit_scale=True,
    #               miniters=1,
    #               desc=eg_link.split('/')[-1]) as t:
    #     urllib.urlretrieve(eg_link, filename=os.devnull,
    #                        reporthook=t.update_to, data=None)
