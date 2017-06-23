import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
import os


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
    

def get_dataset(args):
    if args.dataset == 'floorplans':
        fn = 'data/floorplans.train.tfrecords'
        d = TFRecordDataset(fn)
        d = d.map(parse_floorplans)
    elif args.dataset == 'cifar':
        fn = 'data/cifar.32.train.tfrecords'
        d = TFRecordDataset(fn)
        d = d.map(parse_cifar)

    # d = tf.train.shuffle_batch(d, batch_size=args.batch_size, capacity=10000, num_threads=4, min_after_dequeue=20)
    # x = d
    # d = d.cache('tmp/')
    d = d.cache(os.path.join(args.cache_dir, 'cache')) if args.cache_dir else d.cache()
    d = d.repeat()
    d = d.shuffle(buffer_size=args.buffer_size)
    d = d.batch(args.batch_size * args.n_gpus)

    iterator = d.make_initializable_iterator()
    x = iterator.get_next()

    # determine size of dataset
    c = sum([1 for r in tf.python_io.tf_record_iterator(fn)])

    # return d, int(c)
    
    return x, iterator.initializer, int(c)

        
