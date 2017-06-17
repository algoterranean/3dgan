import tensorflow as tf
import numpy as np
import h5py
import sys
import os



class TFRecordsDataset:
    def __init__(self, filenames, feature_def, image_shape, num_threads=4):
        """Generates a batches input queue from a list TFRecords files and their
        corresponding ProtoBuf def."""
        self.filenames = filenames
        self.num_threads = num_threads
        self.feature_def = feature_def
        self.image_shape = image_shape

    def batch_tensor(self, batch_size, num_epochs): #, normalize=False):
        with tf.name_scope('input_queue'):
            filename_queue = tf.train.string_input_producer(self.filenames, num_epochs=num_epochs)
            image = self._read_and_decode(filename_queue) #, normalize)
            images = tf.train.shuffle_batch([image],
                                                batch_size=batch_size,
                                                num_threads=self.num_threads,
                                                capacity=1*batch_size+1000,
                                                min_after_dequeue=1000)
        return images


    # img_1d = np.fromstring(img_string, dtype=np.uint8)
    

    def _read_and_decode(self, filename_queue): #, normalize=False):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=self.feature_def)
        image = tf.decode_raw(features['image_raw'], tf.uint8)

        # img_string = features

        # img_1d = np.fromstring(img_string, dtype=np.uint8)


        # get size of flattened shape
        c = 1
        for i in self.image_shape:
            c *= i

        image.set_shape([c])
        image = tf.reshape(image, self.image_shape)
        image = tf.cast(image, tf.float32) / 255.0
        # image = image / 255.0
        # image = tf.cast(image, tf.float32) * (1.0 / 255.0)
        # image = tf.image.resize_images(image, [64, 64])
        # image = tf.image.per_image_standardization(image)
        # if normalize:
        #     image = image - 0.5
        return image


    
def get_dataset(name):
    if name == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        return input_data.read_data_sets("data/MNIST_data", one_hot=True)
    elif name == 'floorplans':
        return TFRecordsDataset([os.path.join('data', 'floorplans.64.train.tfrecords')],
                                    {'image_raw': tf.FixedLenFeature([], tf.string)},
                                    [64, 64, 3])
    elif name == 'cifar':
        return TFRecordsDataset([os.path.join('data', 'cifar.32.train.tfrecords')],
                                    {'image_raw': tf.FixedLenFeature([], tf.string)},
                                    [32, 32, 3])
