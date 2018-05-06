import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
import os


def bytes_feature(value):
    if not isinstance(value, type([])):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    if not isinstance(value, type([])):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, type([])):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class DataPlugin:
    def __init__(self):
        self.name = None

    @staticmethod
    def arguments():
        """List of **kwargs suitable for feeding into argparse."""
        args = {'--example_argument': {'type': int,
                                       'nargs': 2,
                                       'help' : """Resize input images to size w x h. This argument, if specified, 
                                       requires two values (width and height)."""}}
        return args


    @staticmethod
    def check_files(storage_dir, required_files):
        """Check that a dir contains all required files."""
        file_list = os.listdir(storage_dir)
        for f in required_files:
            if f not in file_list:
                return False
        return True

    @staticmethod
    def check_prepared_datasets(storage_dir):
        """Check the local filesystem for already prepared tfrecord files."""
        return False

    @staticmethod
    def check_raw_datasets(storage_dir):
        """Check the local filesystem for already downloaded raw dataset files."""
        return False

    @staticmethod
    def download(download_dir):
        return False

    @staticmethod
    def convert_to_tfrecord(download_dir, storage_dir):
        return False

    @staticmethod
    def get_datasets(storage_dir):
        """Return a dictionary containing all available datasets.

        The keys should be one of 'train', 'test', or 'validate',
        and their values in the dict should be tuples, where the first
        entry is the mapped TFRecordDataset, and the second is the number
        of records in the dataset."""
        return {'train': None, 'test': None, 'validate': None}

    @staticmethod
    def _get_dataset(file_name, parser):
        """Helper function. Intended to be used by get_datasets."""
        dataset = TFRecordDataset(file_name).map(parser)
        dataset_size = sum([1 for r in tf.python_io.tf_record_iterator(file_name)])
        return dataset, dataset_size

    @staticmethod
    def parse_tfrecord(example_proto):
        pass
