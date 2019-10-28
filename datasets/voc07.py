import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets import dataset_utils


FILE_PATTERN = 'voc_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (238, 306),
    'bicycle': (243, 353),
    'bird': (330, 486),
    'boat': (181, 290),
    'bottle': (244, 505),
    'bus': (186, 229),
    'car': (713, 1250),
    'cat': (337, 376),
    'chair': (445, 798),
    'cow': (141, 259),
    'diningtable': (200, 215),
    'dog': (421, 510),
    'horse': (287, 362),
    'motorbike': (245, 339),
    'person': (2008, 4690),
    'pottedplant': (245, 514),
    'sheep': (96, 257),
    'sofa': (229, 248),
    'train': (261, 297),
    'tvmonitor': (256, 324),
    'total': (5011, 12608),
}
TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20),
}
SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 20

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

def get_split(split_name, dataset_dir, file_pattern,        #the pattern contains a '%s' string so that the split name can be inserted
              reader, split_to_sizes=SPLITS_TO_SIZES,
              items_to_description=ITEMS_TO_DESCRIPTIONS,
              num_classes=NUM_CLASSES):
    if split_name not in split_to_sizes:
        raise ValueError('split name %s not recognized' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers
    )

    labels_to_name = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_name = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes[split_name],
        items_to_descriptions=items_to_description,
        num_classes=num_classes,
        labels_to_name=labels_to_name
    )