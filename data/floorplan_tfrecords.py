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
    writer = tf.python_io.TFRecordWriter('floorplans.{}.tfrecords'.format(name))

    image_dir = os.path.join('/mnt/research/datasets/floorplans')
    lines = open(os.path.join(image_dir, filename)).readlines()
    for line in tqdm(lines):
        fn = os.path.join(image_dir, line.strip())
        # read image
        with tf.gfile.FastGFile(fn, 'rb') as f:
            image_data = f.read()
        # determine shape
        f = open(fn, 'rb')
        stuff = f.read()
        f.close()
        nparr = np.fromstring(stuff, np.uint8)
        i = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        w, h, c = i.shape
        # write record
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(image_data),
                'width': _int64_feature(w),
                'height': _int64_feature(h),
                'channels': _int64_feature(c),
                'filename': _bytes_feature(tf.compat.as_bytes(fn))}))
        writer.write(example.SerializeToString())
        
if __name__ == '__main__':
    # generate_dataset('train', 'train_set.txt')
    # generate_dataset('test', 'test_set.txt')
    # generate_dataset('validate', 'validation_set.txt')    
    fn = 'floorplans.test.tfrecords'
    c = sum([1 for r in tf.python_io.tf_record_iterator(fn)])
    print('floorplans test:', c)    


