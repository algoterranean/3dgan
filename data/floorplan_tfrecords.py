from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2
import os


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# TODO what to do about "Premature end of JPEG file?"
# Can we catch it? or... ?


def generate_dataset(name, filename):
    writer = tf.python_io.TFRecordWriter('floorplans.64.{}.tfrecords'.format(name))

    image_dir = os.path.join('/mnt/research/datasets/floorplans')
    lines = open(os.path.join(image_dir, filename)).readlines()
    for line in tqdm(lines):
        fn = os.path.join(image_dir, line.strip())
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        
        img_string = img.tostring()
        if len(img_string) != 12288:
            print('Bad image!', fn)
        else:
            img_w = 64
            img_h = 64
            example = tf.train.Example(features=tf.train.Features(feature={'image_raw': _bytes_feature(img_string)}))
            serialized = example.SerializeToString()
            writer.write(serialized)
    
generate_dataset('train', 'train_set.txt')
generate_dataset('test', 'test_set.txt')
generate_dataset('validate', 'validation_set.txt')

