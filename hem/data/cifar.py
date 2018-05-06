import os
import pickle
import urllib

import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from tqdm import tqdm

import hem
# from hem.data.DataPlugin import DataPlugin, _bytes_feature

# TODO what to do about "Premature end of JPEG file?"
# Can we catch it? or... ?


# TODO add categories to dataset

_features = {
    'image': tf.FixedLenFeature([], tf.string)
    }

_output_files = ['cifar.train.tfrecords',
                 'cifar.test.tfrecords']

_input_files = ['cifar-10-python.tar.gz']


class CIFARDataset(hem.DataPlugin):
    name = 'cifar'

    @staticmethod
    def check_prepared_datasets(storage_dir):
        return CIFARDataset.check_files(storage_dir, _output_files)

    @staticmethod
    def check_raw_datasets(storage_dir):
        return CIFARDataset.check_files(storage_dir, _input_files)

    @staticmethod
    def download(download_dir):
        # TODO: add tqdm support. see https://pypi.python.org/pypi/tqdm for example
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        try:
            urllib.request.urlretrieve(url, os.path.join(download_dir, 'cifar-10-python.tar.gz'))
        except Exception as e:
            print(e)
            return False
        return True

    @staticmethod
    def convert_to_tfrecord(download_dir, storage_dir):
        def build_dataset(name, filelist):
            writer = tf.python_io.TFRecordWriter('cifar.{}.tfrecords'.format(name))
            # image_dir = os.path.join(download_dir) #'/mnt/research/datasets/cifar-10/cifar-10-batches-py')

            for x in filelist:
                cifar_dict = pickle.load(open(os.path.join(download_dir, x), 'rb'), encoding='bytes')
                images = cifar_dict[b'data']
                num_images = images.shape[0]
                images = images.reshape((num_images, 3, 32, 32))
                images = images.transpose((0, 2, 3, 1))

                for img in tqdm(images):
                    img_string = img.tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(feature={'image': hem.bytes_feature(img_string)}))
                    writer.write(example.SerializeToString())
        build_dataset('train', ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
        build_dataset('test', ['test_batch'])

    @staticmethod
    def get_datasets(args):
        train_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[0]))
        test_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[2]))
        train_set = train_set.map(CIFARDataset.parse_tfrecord(args))
        test_set = test_set.map(CIFARDataset.parse_tfrecord(args))
        return {'train': train_set, 'test': test_set, 'validate': None}

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
            image = tf.transpose(image, [2, 0, 1])
            return image
        return helper
