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
    writer = tf.python_io.TFRecordWriter('nyuv2.{}.tfrecords'.format(name))

    image_dir = os.path.join('/mnt/research/datasets/nyuv2/preprocessed')
    lines = open(os.path.join(image_dir, filename)).readlines()
    for line in tqdm(lines):
        fin = os.path.join(image_dir, line.strip() + '_i.png')
        fdn = os.path.join(image_dir, line.strip() + '_f.png')
        with tf.gfile.FastGFile(fin, 'rb') as f:
            image_data = f.read()
        with tf.gfile.FastGFile(fdn, 'rb') as f:
            depth_data = f.read()

        # depth_data = cv2.imread(fdn, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # print(depth_data.min(), depth_data.max())
        # depth_data = np.divide(depth_data, np.iinfo(np.uint16).max)
        # print(depth_data.min(), depth_data.max())

        w, h, c = 427, 561, 3
            
        # # determine shape
        # f = open(fn, 'rb')
        # stuff = f.read()
        # f.close()
        # nparr = np.fromstring(stuff, np.uint8)
        # i = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # w, h, c = i.shape
        
        # write record
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(image_data),
                'depth': _bytes_feature(depth_data),
                'width': _int64_feature(w),
                'height': _int64_feature(h),
                'channels': _int64_feature(c),
                'filename': _bytes_feature(tf.compat.as_bytes(fin)),
                'depth_filename': _bytes_feature(tf.compat.as_bytes(fdn))
                }))
        writer.write(example.SerializeToString())
        
    
# generate_dataset('train', 'train.txt')
# generate_dataset('test', 'test.txt')
generate_dataset('validate', 'validation.txt')

