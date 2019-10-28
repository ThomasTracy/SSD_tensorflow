import os
import sys
import random

import tensorflow as tf
import xml.etree.ElementTree as ET
from datasets.voc07 import VOC_LABELS
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature


RANDOM_SEED = 1314
SAMPLES_PER_FILES = 200

DIR_IMAGES = 'JPEGImages'
DIR_ANNOTATIONS = 'Annotations'


def _process_image(dir, name):

    # Get file
    filename = os.path.join(dir, DIR_IMAGES, name+'.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    # Get annotation
    filename = os.path.join(dir, DIR_ANNOTATIONS, name+'.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((
            float(bbox.find('ymin').text) / shape[0],
            float(bbox.find('xmin').text) / shape[1],
            float(bbox.find('ymax').text) / shape[0],
            float(bbox.find('xmax').text) / shape[1]
        ))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert2example(image_data, shape, bboxes, labels,
                     labels_text, difficult, truncated):

    y_min = []
    x_min = []
    y_max = []
    x_max = []
    for b in bboxes:
        assert len(b) == 4
        [xy.append(point) for xy, point in zip([y_min, x_min, y_max, x_max], b)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(x_min),
        'image/object/bbox/xmax': float_feature(x_max),
        'image/object/bbox/ymin': float_feature(y_min),
        'image/object/bbox/ymax': float_feature(y_max),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))
    return example


def _add2tfrecord(dataset_dir, name, tfrecord_writer):

    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert2example(image_data, shape, bboxes, labels,
                               labels_text, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    if not tf.gfile.Exists(dataset_dir):
        raise ValueError('Dictionary %s doesnt exist' %dataset_dir)

    path = os.path.join(dataset_dir, DIR_IMAGES)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Precess dataset files
    i = 0
    findx = 0
    while(i < len(filenames)):

        tf_filename = '%s/%s_%03d.tfrecord' % (output_dir, name, findx)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while(i<len(filenames) and j<SAMPLES_PER_FILES):
                sys.stdout.write('\r>> Converting image %d/%d' %(i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename.split('.')[0]
                _add2tfrecord(dataset_dir, img_name, writer)
                i += 1
                j += 1
            findx += 1

    print("\rFinished converting VOC dataset")


if __name__ == "__main__":
    dataset_dir = 'D:\Data\VOCdevkit\VOC2012'
    output_dir = 'D:\Data\VOC\\train'
    run(dataset_dir, output_dir)
