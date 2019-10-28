from PIL import Image
from tools import anchors_show
from matplotlib import pyplot as plt
from enum import Enum, IntEnum
import numpy
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import custom_layers, ssd_vgg
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import device_lib
from preprocessing import ssd_vgg_preprocessing
from tools import show_tfrecord
from train import FLAGS
from nets import ssd_vgg


def run():
    dataset = show_tfrecord.get_from_tfrecord()
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size,
        shuffle=True
    )
    preprocess_fun = ssd_vgg_preprocessing.preprocess_image

    [image_org, shape_org, bbox_org, label_org] = provider.get(['image', 'shape', 'object/bbox', 'object/label'])
    image, label, bbox = preprocess_fun(image_org, label_org,
                                               bbox_org, out_shape=(512, 512),
                                               data_format='NCHW',
                                               is_training=True)

    ssd_class = ssd_vgg.SSDNet
    ssd_param = ssd_class.default_parameters._replace(num_classes=FLAGS.num_classes)
    print("Class numbers", ssd_params.num_classes)
    ssd_net = ssd_class(ssd_param)
    ssd_shape = ssd_net.params.img_shape
    ssd_anchors =


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image, label, bbox= sess.run([image, label, bbox])

        image = numpy.transpose(image, (1, 2, 0))
        show_tfrecord.show_one_image([image],
                                     [bbox],
                                     [label])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    run()