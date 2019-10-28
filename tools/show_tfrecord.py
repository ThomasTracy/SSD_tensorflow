from datasets import voc07
from datasets import dataset_factory
from train import FLAGS
from matplotlib import pyplot as plt
from preprocessing import ssd_vgg_preprocessing

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy


def read_tfrecord(input_file):
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
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    items_to_discriptions = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/bbox': 'A list of bounding boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }


def get_from_tfrecord():

    return dataset_factory.get_dataset('pascalvoc_2007', 'train', 'D:\Data\VOC\\train')


def show_one_image(image, bboxes, labels):
    if not isinstance(image, list):
        raise ValueError('Please wrappe inputs in list first')
    num_subplot = len(image)
    plt.figure('Image')

    for i in range(num_subplot):
        plt.subplot(1, num_subplot, i+1)
        plt.imshow(image[i])
        shape = [image[i].shape[0], image[i].shape[1]]
        bbox = bboxes[i]
        label = labels[i]
        for ([y_min, x_min, y_max, x_max], l) in zip(bbox, label):
            x_draw = x_min * shape[1]
            y_draw = y_min * shape[0]
            w_draw = (x_max - x_min) * shape[1]
            h_draw = (y_max - y_min) * shape[0]
            plt.gca().add_patch(plt.Rectangle((x_draw, y_draw), w_draw, h_draw,
                                              edgecolor='r', linewidth=1, facecolor='None'))
            plt.text(x_draw + w_draw/2, y_draw, l)

    plt.show()


def run():
    dataset = get_from_tfrecord()
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size,
        shuffle=True
    )
    preprocess_fun = ssd_vgg_preprocessing.preprocess_image

    [image_org, shape_org, bbox_org, label_org] = provider.get(['image', 'shape', 'object/bbox', 'object/label'])
    # image, labels, bboxes = preprocess_fun(image_org, label_org,
    #                                        bbox_org, out_shape=(300, 300),
    #                                        data_format='NCHW',
    #                                        is_training=True)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        '''此处image，bbox等一定要在同一个线程中同时取出。若分别通过不同的sess.run取出，
        会多次激活provider.get,而每次激活时shuffle又为True，将导致image，bbox，shape等
        不匹配，混乱。
        '''
        image_org, shape_org, bbox_org, label_org =  sess.run([image_org, shape_org, bbox_org, label_org])
        # image_show, shape_org, bbox_show, label_show = sess.run([image, shape_org, bboxes, labels])
        image_org_tensor = tf.convert_to_tensor(image_org, image_org.dtype)
        label_org_tensor = tf.convert_to_tensor(label_org, label_org.dtype)
        bbox_org_tensor = tf.convert_to_tensor(bbox_org, bbox_org.dtype)
        image_show, label_show, bbox_show = preprocess_fun(image_org_tensor, label_org_tensor,
                                               bbox_org_tensor, out_shape=(512, 512),
                                               data_format='NCHW',
                                               is_training=True)
        image_show, label_show, bbox_show = sess.run([image_show, label_show, bbox_show])
        image_show = numpy.transpose(image_show, (1, 2, 0))

        print(image_show.shape)
        print(label_show, label_org)
        print(bbox_show, bbox_org)
        show_one_image([image_org, image_show],
                       [bbox_org, bbox_show],
                       [label_org, label_show])
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    run()