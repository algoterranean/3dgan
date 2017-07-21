import tensorflow as tf
import numpy as np
import urllib
import os
import struct
import gzip
from data.DataPlugin import DataPlugin, _bytes_feature, _int64_feature


_required_files = ['t10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz',
                   'train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz']

_features = {
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
    }


class TestDataset(DataPlugin):
    def __init__(self):
        self.name = 'mnist'

    def check_local(self, storage_dir):
        file_list = os.listdir(storage_dir)
        for f in _required_files:
            if f not in file_list:
                return False
        return True

    def download(self, download_dir):
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        try:
            for f in _required_files:
                urllib.urlretrieve(base_url + f, os.path.join(download_dir, f))
        except Exception as e:
            print(e)
            return False
        return True

    def convert_to_tfrecord(self, download_dir, storage_dir):
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
                    'image': _bytes_feature(img),
                    'label': _int64_feature(label)
                    }))
                writer.write(record.SerializeToString())
            writer.close()

        write_tfrecord(read_images(_required_files[0]),
                       read_labels(_required_files[1]),
                       'mnist.test.tfrecords')
        write_tfrecord(read_images(_required_files[2]),
                       read_labels(_required_files[3]),
                       'mnist.train.tfrecords')

    def parse_tfrecord(self, example_proto):
        parsed = tf.parse_single_example(example_proto, _features)
        # decode image, shape it, and convert to [0, 1]
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, [28, 28, 1])
        image = tf.cast(image, tf.float32) / 255.0
        # label
        label = parsed['label']
        return image, label


if __name__ == '__main__':
    from tensorflow.contrib.data import TFRecordDataset

    loc = '/Users/p/Desktop/autoencoders/data/MNIST_data'
    loc2 = '/Users/p/Desktop/autoencoders/data/test_data'
    d = TestDataset()
    # d.convert_to_tfrecord(loc, loc2)

    d2 = TFRecordDataset(os.path.join(loc2, 'mnist.test.tfrecords')).map(d.parse_tfrecord)
    iterator = d2.make_initializable_iterator()
    x = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        result = sess.run(x)
        print(result[0].shape, result[1])






