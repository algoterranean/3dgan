from tqdm import tqdm
import tensorflow as tf
import os
import pickle


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# TODO what to do about "Premature end of JPEG file?"
# Can we catch it? or... ?


def generate_dataset(name, filenames):
  writer = tf.python_io.TFRecordWriter('cifar.32.{}.tfrecords'.format(name))
  image_dir = os.path.join('/mnt/research/datasets/cifar-10/cifar-10-batches-py')

  for x in filenames:
    dict = pickle.load(open(os.path.join(image_dir,x), 'rb'), encoding='bytes')
    images = dict[b'data']
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))
        
    for img in tqdm(images):
      img_string = img.tostring()
      example = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(img_string)}))
      writer.write(example.SerializeToString())
        

if __name__ == '__main__':
  generate_dataset('train', ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
  generate_dataset('test', ['test_batch'])
    
  # fn = 'cifar.train.tfrecords'
  # c = sum([1 for r in tf.python_io.tf_record_iterator(fn)])
  # print('cifar10 train:', c)



