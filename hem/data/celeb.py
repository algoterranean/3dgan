import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset

import hem
# from hem.data.DataPlugin import DataPlugin, _bytes_feature

# 5_o_Clock_Shadow
# Arched_Eyebrows
# Attractive
# Bags_Under_Eyes
# Bald
# Bangs
# Big_Lips
# Big_Nose
# Black_Hair
# Blond_Hair
# Blurry
# Brown_Hair
# Bushy_Eyebrows
# Chubby
# Double_Chin
# Eyeglasses
# Goatee
# Gray_Hair
# Heavy_Makeup
# High_Cheekbones
# Male
# Mouth_Slightly_Open
# Mustache
# Narrow_Eyes
# No_Beard
# Oval_Face
# Pale_Skin
# Pointy_Nose
# Receding_Hairline
# Rosy_Cheeks
# Sideburns
# Smiling
# Straight_Hair
# Wavy_Hair
# Wearing_Earrings
# Wearing_Hat
# Wearing_Lipstick
# Wearing_Necklace
# Wearing_Necktie
# Young


_features = {
    'image': tf.FixedLenFeature([], tf.string),
    'filename': tf.FixedLenFeature([], tf.string),
    'attributes': tf.FixedLenFeature([], tf.string)
    }

_output_files = ['celeba.train.tfrecords',
                 'celeba.validate.tfrecords',
                 'celeba.test.tfrecords']

_input_files = ['img_align_celeba.zip']


class CelebDataset(hem.DataPlugin):
    name = 'celeb'

    # celeba aligned dataset = 178 x 218

    # TODO this isn't working? always returns False
    @staticmethod
    def check_prepared_datasets(storage_dir):
        CelebDataset.check_files(storage_dir, _output_files)

    # TODO this isn't working? always returns False
    @staticmethod
    def check_raw_datasets(storage_dir):
        CelebDataset.check_files(storage_dir, _input_files)

    @staticmethod
    def download(download_dir):
        pass

    @staticmethod
    def convert_to_tfrecord(download_dir, storage_dir):
        png_dir = os.path.join(download_dir, 'img_align_celeba_png')
        jpg_dir = os.path.join(download_dir, 'img_align_celeba_jpg')
        partition = open(os.path.join(download_dir, 'list_eval_partition.txt'), 'r')
        training_images = []
        validation_images = []
        testing_images = []

        for fn in partition.readlines():
            image_fn, which = fn.strip().split()
            if which == "0":
                training_images.append(image_fn)
            elif which == "1":
                validation_images.append(image_fn)
            elif which == "2":
                testing_images.append(image_fn)

        attributes = {}
        af = open(os.path.join(download_dir, 'list_attr_celeba.txt'))
        for line in af.readlines()[2:]:
            d = line.strip().split()
            attribute_list = [True if x == '1' else False for x in d[1:]]
            attributes[d[0]] = np.array(attribute_list)
            # jj = attributes[d[0]].tostring()
            # kk = np.fromstring(jj, dtype=np.bool)

        def build_dataset(name, filelist):
            writer = tf.python_io.TFRecordWriter(os.path.join(storage_dir, 'celeba.{}.tfrecords'.format(name)))
            for img_fn in filelist:
                path = os.path.join(png_dir, img_fn) if img_fn.endswith('.png') else os.path.join(jpg_dir, img_fn)
                with tf.gfile.FastGFile(path, 'rb') as f:
                    image_data = f.read()
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': hem.bytes_feature(image_data),
                        'filename': hem.bytes_feature(tf.compat.as_bytes(img_fn)),
                        'attributes': hem.bytes_feature(attributes[img_fn].tostring())
                        }))
                writer.write(example.SerializeToString())

        build_dataset('test', testing_images)
        build_dataset('validate', validation_images)
        build_dataset('train', training_images)

    @staticmethod
    def parse_tfrecord(args):
        def helper(example_proto):
            # TODO resize shoudl be a parameter, not hard-coded
            parsed = tf.parse_single_example(example_proto, _features)
            image = tf.image.decode_image(parsed['image'], channels=3)
            image = tf.reshape(image, [178, 218, 3])
            image = tf.image.resize_images(image, [64, 64])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.transpose(image, [2, 0, 1])
            return image
        return helper

    @staticmethod
    def get_datasets(args):
        train_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[0]))
        validate_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[1]))
        test_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[2]))
        train_set = train_set.map(CelebDataset.parse_tfrecord(args))
        validate_set = validate_set.map(CelebDataset.parse_tfrecord(args))
        test_set = test_set.map(CelebDataset.parse_tfrecord(args))
        return {'train': train_set,  'validate': validate_set, 'test': test_set}


if __name__ == '__main__':
    # TODO need to test out datasets fully, including downloading AND unzipping AND generating splits
    # ensure that the dataset exists
    p = CelebDataset()
    tfrecord_dir = '/mnt/research/projects/autoencoders/data/storage'
    raw_dir = '/mnt/research/datasets/celeba'
    if not p.check_prepared_datasets(tfrecord_dir):
        if not p.check_raw_datasets(raw_dir):
            print('Downloading dataset...')
            # TODO: datasets should be able to be marked as non-downloadable
            p.download(raw_dir)
        print('Converting to tfrecord...')
        p.convert_to_tfrecord(raw_dir, tfrecord_dir)

    # # load the dataset
    # datasets = p.get_datasets(tfrecord_dir)
    # prepared_datasets = {}
