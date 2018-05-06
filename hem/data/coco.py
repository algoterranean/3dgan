import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import urllib

import hem

# from hem.data.DataPlugin import *

_features = {
    'image': tf.FixedLenFeature([], tf.string),
    # all masks combined into one image
    'annotations': tf.FixedLenFeature([], tf.string),
    'width': tf.FixedLenFeature([], tf.int64),
    'height': tf.FixedLenFeature([], tf.int64),
    'filename': tf.FixedLenFeature([], tf.string),
    'image_id': tf.FixedLenFeature([], tf.int64),
    # annotation information
    'labels': tf.VarLenFeature(tf.int64),
    'bboxes': tf.VarLenFeature(tf.float32),
    'iscrowds': tf.VarLenFeature(tf.int64),
    'areas': tf.VarLenFeature(tf.float32)
    }

_output_files = ['coco.train.tfrecords',
                 'coco.validate.tfrecords',
                 'coco.test.tfrecords']


class COCODataset(hem.DataPlugin):
    name = 'coco'

    @staticmethod
    def download(download_dir):
        base_url = 'http://msvocds.blob.core.windows.net/'
        # images
        image_files = ['coco2014/train2014.zip',
                       'coco2014/val2014.zip',
                       'coco2014/test2014.zip',
                       'coco2015/test2015.zip']
        # annotations
        annotation_files = ['annotations-1-0-3/instances_train-val2014.zip',        # obj instances
                            'annotations-1-0-3/person_keypoints_trainval2014.zip',  # person keypoints
                            'annotations-1-0-3/captions_train-val2014.zip',         # image captions
                            'annotations-1-0-4/image_info_test2014.zip',            # testing image info
                            'annotations-1-0-4/image_info_test2015.zip']            # testing image info
        try:
            for f in image_files + annotation_files:
                urllib.request.urlretrieve(base_url + f, os.path.join(download_dir, f))
        except Exception as e:
            raise e
        return True

    @staticmethod
    def check_prepared_datasets(storage_dir):
        for f in _output_files:
            if not os.path.exists(os.path.join(storage_dir, f)):
                return False
        return True

    @staticmethod
    def check_raw_datasets(storage_dir):
        for f in ['test2014.zip', 'instances_train-val2014.zip']:
            if not os.path.exists(os.path.join(storage_dir, f)):
                return False
        return True

    @staticmethod
    def convert_to_tfrecord(download_dir, storage_dir):
        def build_dataset(name):
            image_dirs = {'train': 'train2014',
                          'validate': 'val2014',
                          'test': 'test2014'}
            annotate_files = {'train': 'instances_train2014.json',
                              'validate': 'instances_val2014.json',
                              'test': 'image_info_test2014.json'}
            image_dir = os.path.join(download_dir, image_dirs[name])
            annotate_file = os.path.join(download_dir, 'annotations', annotate_files[name])

            coco = COCO(annotate_file)
            categories = coco.loadCats(coco.getCatIds())
            category_names = set([c['name'] for c in categories])
            category_ids = coco.getCatIds(catNms=category_names)
            image_ids = coco.getImgIds(catIds=category_ids)

            file_list = []
            for i in image_ids:
                # keys = license, file_name, coco_url, height, width, date_captured, flickr_url, id
                img = coco.loadImgs(i)[0]
                file_list.append(img)

            writer = tf.python_io.TFRecordWriter(os.path.join(storage_dir, 'coco.{}.tfrecords'.format(name)))

            for img in tqdm(file_list):
                path = os.path.join(image_dir, img['file_name'])
                with tf.gfile.FastGFile(path, 'rb') as f:
                    image_data = f.read()

                # create mask
                annotations = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
                total_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)
                label_list = []
                bbox_list = []
                crowd_list = []
                area_list = []
                for a in annotations:
                    mask = coco.annToMask(a)
                    total_mask[mask == 1] = int(a['category_id'])
                    for b in a['bbox']:
                        bbox_list.append(b)
                    # bbox_list.append(a['bbox'])
                    crowd_list.append(a['iscrowd'])
                    area_list.append(a['area'])
                    label_list.append(a['category_id'])

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': hem.bytes_feature(image_data),
                        'annotations': hem.bytes_feature(total_mask.tostring()),
                        'filename': hem.bytes_feature(tf.compat.as_bytes(img['file_name'])),
                        'width': hem.int64_feature(img['width']),
                        'height': hem.int64_feature(img['height']),
                        'image_id': hem.int64_feature(img['id']),
                        'bboxes': hem.float_feature(bbox_list),
                        'iscrowds': hem.int64_feature(crowd_list),
                        'areas': hem.float_feature(area_list),
                        'labels': hem.int64_feature(label_list)
                        }))
                writer.write(example.SerializeToString())

        build_dataset('train')
        build_dataset('validate')
        build_dataset('test')

    @staticmethod
    def parse_tfrecord(args):
        def helper(example_proto):
            # TODO resize should be a parameter, not hard-coded
            parsed = tf.parse_single_example(example_proto, _features)
            image = tf.image.decode_image(parsed['image'], channels=3)
            w = parsed['width']
            h = parsed['height']
            image = tf.reshape(image, [h, w, 3])
            image = tf.image.resize_images(image, [64, 64])
            image = tf.cast(image, tf.float32) / 255.0
            annotations = tf.image.decode_image(parsed['annotations'], channels=1)
            annotations = tf.reshape(annotations, [h, w, 1])
            image = tf.transpose(image, [2, 0, 1])
            annotations = tf.transpose(annotations, [2, 0, 1])
            return image, annotations
        return helper

    @staticmethod
    def get_datasets(args):
        train_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[0]))
        validate_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[1]))
        test_set = TFRecordDataset(os.path.join(args.dataset_dir, _output_files[2]))
        train_set = train_set.map(COCODataset.parse_tfrecord(args))
        validate_set = validate_set.map(COCODataset.parse_tfrecord(args))
        test_set = test_set.map(COCODataset.parse_tfrecord(args))
        return {'train': train_set,  'validate': validate_set, 'test': test_set}


if __name__ == '__main__':
    # TODO need to test out datasets fully, including downloading AND unzipping AND generating splits
    # ensure that the dataset exists
    p = COCODataset()
    tfrecord_dir = '/mnt/research/projects/autoencoders/data/storage'
    raw_dir = '/mnt/research/datasets/coco'
    if not p.check_prepared_datasets(tfrecord_dir):
        if not p.check_raw_datasets(raw_dir):
            print('Downloading dataset...')
            # TODO: datasets should be able to be marked as non-downloadable
            p.download(raw_dir)
        print('Converting to tfrecord...')
        p.convert_to_tfrecord(raw_dir, tfrecord_dir)
