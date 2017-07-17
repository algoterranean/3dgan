import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import TFRecordDataset
import os


TRAIN = 0
VALIDATE = 1
TEST = 2

def parse_floorplans(example_proto):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
        'filename': tf.FixedLenFeature([], tf.string) }

    parsed = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_image(parsed['image'], channels=3)
    w = tf.cast(parsed['width'], tf.int32)
    h = tf.cast(parsed['height'], tf.int32)
    c = tf.cast(parsed['channels'], tf.int32)
    image_shape = tf.stack([w, h, c])
    image = tf.reshape(image, image_shape)
    image = tf.image.resize_images(image, [64, 64])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def parse_cifar(example_proto):
    features = {'image': tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features)
    image = parsed['image']
    image = tf.cast(image, tf.float32) / 255.0
    return image


def parse_nyuv2(example_proto):
    # dims = 427, 561, 3
    # 142.3 x 187,
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'depth': tf.FixedLenFeature([], tf.string),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
        'filename': tf.FixedLenFeature([], tf.string),
        'depth_filename': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_image(parsed['image'], channels=3)
    depth = tf.image.decode_png(parsed['depth'], channels=1, dtype=tf.uint16)
    w = tf.cast(parsed['width'], tf.int32)
    h = tf.cast(parsed['height'], tf.int32)
    c = tf.cast(parsed['channels'], tf.int32)
    image_shape = tf.stack([w, h, c])
    image = tf.reshape(image, image_shape)
    # image = tf.image.resize_images(image, [142, 187]) #  x3
    image = tf.image.resize_images(image, [64, 64]) #  x3    
    image = tf.cast(image, tf.float32) / 255.0
    
    depth = tf.reshape(depth, tf.stack([w, h, 1]))
    depth = tf.cast(depth, tf.float32) / np.iinfo(np.uint16).max
    depth = tf.image.resize_images(depth, [64, 64]) # x3
    # depth = tf.image.resize_images(depth, [142, 187]) # x3

    
    # depth = tf.cast(depth, tf.float32) #/ np.iinfo(np.uint16).max)

    return tf.concat((image, depth), axis=2)
    # return (image, depth)





def get_dataset(args, which=TRAIN):

    if args.dataset == 'floorplans':
        # train = 160830
        # validate = 32166
        # test = 128666
        fn = {TRAIN: 'data/floorplans.train.tfrecords',
                  VALIDATE: 'data/floorplans.validate.tfrecords',
                  TEST: 'data/floorplans.test.tfrecords'}[which]
        d = TFRecordDataset(fn)        
        d = d.map(parse_floorplans)
    elif args.dataset == 'cifar':
        # cifar10 has no validation set
        # train = 50000
        # test = 10000
        fn = {TRAIN: 'data/cifar.32.train.tfrecords',
                  VALIDATE: 'data/cifar.32.test.tfrecords',
                  TEST: 'data/cifar.32.test.tfrecords'}[which]
        d = TFRecordDataset(fn)
        d = d.map(parse_cifar)
    elif args.dataset == 'nyuv2':
        # train = 27858
        # validate = 3095
        # test = 20343
        c = 27858
        fn = {TRAIN: 'data/nyuv2.train.tfrecords',
                  VALIDATE: 'data/nyuv2.validate.tfrecords',
                  TEST: 'data/nyuv2.test.tfrecords'}[which]
        d = TFRecordDataset(fn)
        d = d.map(parse_nyuv2)
        
    # cache dataset if requested
    cache_fn = {TRAIN: 'cache.train', VALIDATE: 'cache.validate', TEST: 'cache.validate'}[which]
    d = d.cache(os.path.join(args.cache_dir, cache_fn)) if args.cache_dir else d.cache()
    # keep cycling through dataset without limit
    d = d.repeat()
    # shuffle (TODO: should be before repeat?)
    d = d.shuffle(buffer_size=args.buffer_size)
    # batch it up
    d = d.batch(args.batch_size * args.n_gpus)
    # create iterator
    iterator = d.make_initializable_iterator()
    x = iterator.get_next()
    # determine size of dataset
    c = sum([1 for r in tf.python_io.tf_record_iterator(fn)])
    
    return x, iterator.initializer, int(c)

        
