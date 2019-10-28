import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim

from enum import Enum, IntEnum
from tensorflow.python.ops import control_flow_ops
from nets import ssd_common
from Utils import bbox as bbox_util
from Utils import image as image_util


Resize = IntEnum('Resize', ('NONE',
                            'CENTRAL_CROP',     # Crop (and pad if necessary)       value=2
                            'PAD_AND_RESIZE',   # Pad, and resize to output shape.  value=3
                            'WARP_RESIZE'))     # Warp resize.                      value=4

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (512, 512)


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    if image.get_shape().ndims != 3:
        raise ValueError('The size of input image is wrong')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError("Dimensions mismatch between means and image")

    mean = tf.constant(means, dtype=image.dtype)
    image = image-mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    img = numpy.copy(image)
    img += numpy.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(numpy.uint8)
    return img


def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    if unwhitened:
        image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_bboxes = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_bboxes)


def apply_with_random_selector(x, func, num_case):
    sel = tf.random_uniform([], maxval=num_case, dtype=tf.int32)
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_case)
    ])


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.5, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    with tf.name_scope(scope, 'distorted_bbox_crop', [image, bboxes]):
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True
        )
        distort_bbox = distort_bbox[0, 0]

        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])                #The slice lost 3rd dimension

        bboxes = bbox_util.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = bbox_util.bboxes_filter_overlap(labels, bboxes,
                                                         threshold=BBOX_CROP_OVERLAP,
                                                         assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox


def preprocess_for_train(image, labels, bboxes,
                         out_shape, data_format='NHWC',
                         scope='ssd_preprocessing_train'):
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('The size of input image is wrong')
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        tf_summary_image(image, bboxes, 'image_with_bboxes')

        # distort image and bboxes
        dis_image = image
        dis_image, labels, bboxes, distort_bbox = \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        min_object_covered=MIN_OBJECT_COVERED,
                                        aspect_ratio_range=CROP_RATIO_RANGE)
        # Resize image to output size
        dis_image = image_util.resize_image(dis_image, out_shape,
                                            method=tf.image.ResizeMethod.BILINEAR,
                                            align_corners=False)
        tf_summary_image(dis_image, bboxes, 'image_shape_distorted')

        # Randomly flip the image horizontally
        dis_image, bboxes = image_util.random_flip_left_right(dis_image, bboxes)

        # Randonly distort color
        apply_with_random_selector(
            dis_image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_case=4
        )
        tf_summary_image(dis_image, bboxes, 'image_color_distorted')

        # Rescale to VGG input scale
        image = dis_image * 255.
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))     # Question here
        return image, labels, bboxes


def preprocess_image(image,
                     labels,
                     bboxes,
                     out_shape,
                     data_format,
                     is_training=False,
                     **kwargs):
    if is_training:
        return preprocess_for_train(image, labels, bboxes,
                                    out_shape=out_shape,
                                    data_format=data_format)
    else:
        return preprocess_for_eval(image, labels, bboxes,
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)