import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DataPlugin:
    def __init__(self):
        self.name = None

    def check_local(self, storage_dir):
        pass

    def download(self, download_dir):
        pass

    def convert_to_tfrecord(self, download_dir, storage_dir):
        pass

    def parse_tfrecord(self, example_proto):
        pass