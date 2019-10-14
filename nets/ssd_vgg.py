import math
import numpy
from Utils import bbox
from nets import custom_layers, ssd_common
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple


#------------------------------------------------
#              Definition of SSD Net
#------------------------------------------------
SSD_Parameters = namedtuple('SSD_Parameters', ['img_shape',
                                               'num_classes',
                                               'no_anno_lable',
                                               'feat_layers',
                                               'feat_shapes',
                                               'anchor_size_bounds',
                                               'anchor_sizes',
                                               'anchor_ratios',
                                               'anchor_steps',
                                               'anchor_offsets',
                                               'normalizations',
                                               'prior_scaling'])

class SSDNet(object):
    '''
    Default feature layers with size 512 input:
    block1 : (0, 512, 512, 64)
    block2 : (0, 256, 256, 128)
    block3 : (0, 128, 128, 256)
    block4 : (0, 64, 64, 512)
    block5 : (0, 32, 32, 512)
    block6 : (0, 30, 30, 1024)
    block7 : (0, 30, 30, 1024)
    block8 : (0, 15, 15, 512)
    block9 : (0, 8, 8, 256)
    block10 : (0, 4, 4, 256)
    block11 : (0, 2, 2, 256)
    block12 : (0, 1, 1, 256)
    '''
    default_parameters = SSD_Parameters(
        img_shape=(512, 512),
        num_classes=2,
        no_anno_lable=2,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12'],
        feat_shapes=[(64, 64), (30, 30), (15, 15), (8, 8), (4, 4), (2, 2), (1, 1)],
        anchor_size_bounds=[0.1, 0.9],
        anchor_sizes=[
            (21.0, 51.0),
            (51.0, 133.0),
            (133.0, 215.0),
            (215.04, 296.96),
            (296.96, 378.88),
            (378.88, 460.8),
            (460.8, 542.72)
        ],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 128, 256, 512],
        anchor_offsets=0.5,
        normalizations=[20, -1, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, param=None):
        '''
        Use the default params when not given
        '''
        if isinstance(param, SSD_Parameters):
            self.params = param
        else:
            self.params = SSDNet.default_parameters

    def net(self, inputs,
            is_training=True,
            update_feat_shape=True,
            dropout_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):

        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feature_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_prob=dropout_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope
                    )

        '''
        The feat_shape might be different with default definition
        So that it is necessary to update
        '''
        if update_feat_shape:
            shapes = ssd_feat_shape_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def update_feat_shape(self, predictions):
        shapes = ssd_feat_shape_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=numpy.float32):
        return ssd_anchor_all_layers(img_shape=img_shape,
                                     layer_shape=self.params.feat_layers,
                                     anchor_sizes=self.params.anchor_sizes,
                                     anchor_ratios=self.params.anchor_ratios,
                                     anchor_steps=self.params.anchor_steps,
                                     offsets=self.params.anchor_offsets,
                                     dtype=dtype)

    def bboxes_encode(self, labels, bboxes, anchors, scope=None):
        return ssd_common.ssd_bboxes_encode(labels, bboxes, anchors,
                                            self.params.num_classes,
                                            self.params.no_anno_lable,
                                            gt_threshold=0.5,
                                            prior_scaling=self.params.prior_scaling,
                                            scope=scope)

    def detected_bboxes(self, predictions, locations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        #得到对应类别的得分值以及bbox
        rscores, rbboxes = \
            ssd_common.ssd_bboxes_select(predictions, locations,
                                         select_threshold=select_threshold,
                                         num_classes=self.params.num_classes)
        # 按照得分高低，筛选出400个bbox和对应得分
        rscores, rbboxes = \
            bbox.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # 应用非极大值抑制，筛选掉与得分最高bbox重叠率大于0.5的，保留200个
        rscores, rbboxes = \
            bbox.bboxes_nms_batch(rscores, rbboxes,
                                  nms_threshold=nms_threshold,
                                  keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = bbox.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

#-------------------------------------------------
#                  Some Tools
#-------------------------------------------------

def ssd_feat_shape_from_net(predictions, default_shapes=None):      #Get feature maps' shape form prediction tensors
    feat_shape = []
    for pre in predictions:
        shape = pre.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shape.append(shape)
    return feat_shape


def ssd_anchor_one_layer(
        img_shape,
        layer_shape,
        anchor_size,
        anchor_ratio,
        anchor_step,
        offset,
        dtype
        ):
    y, x = numpy.mgrid[0:layer_shape[0], 0:layer_shape[1]]
    y = (y.astype(dtype) + offset) * anchor_step / img_shape[0]     # y: [[0,0,0...0],...[63,63,63...63]]
    x = (x.astype(dtype) + offset) * anchor_step / img_shape[1]     # x: [[0,1,2...63],...[0,1,2...63]]

    y = numpy.expand_dims(y, axis=-1)
    x = numpy.expand_dims(x, axis=-1)

    assert len(anchor_size) > 1, 'There must be at least 2 anchor sizes'
    num_anchors = len(anchor_size) + len(anchor_ratio)
    h = numpy.zeros((num_anchors,), dtype=dtype)
    w = numpy.zeros((num_anchors,), dtype=dtype)
    h[0] = anchor_size[0]/img_shape[0]
    w[0] = anchor_size[0]/img_shape[1]
    h[1] = math.sqrt(anchor_size[0] * anchor_size[1]) / img_shape[0]
    w[1] = math.sqrt(anchor_size[0] * anchor_size[1]) / img_shape[1]
    for i, r in enumerate(anchor_ratio):
        h[i + 2] = anchor_size[0] / img_shape[0] / math.sqrt(r)
        w[i + 2] = anchor_size[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

def ssd_anchor_all_layers(              #Generate proposed anchors for alllayers
        img_shape,
        layer_shape,
        anchor_sizes,
        anchor_ratios,
        anchor_steps,
        offsets=0.5,
        dtype=numpy.float32
        ):
    all_anchors = []
    for i, s in enumerate(layer_shape):
        anchors_one_layer = ssd_anchor_one_layer(img_shape,
                                                 s,
                                                 anchor_sizes[i],
                                                 anchor_ratios[i],
                                                 anchor_steps[i],
                                                 offsets=offsets,
                                                 dtype=dtype)
        all_anchors.append(anchors_one_layer)
    return all_anchors

#-------------------------------------------------
#               Fuctions of SSD Net
#-------------------------------------------------

def tensor_shape(tensor, rank=3):
    if tensor.get_shape().is_fully_defined():
        print("It is fully defined")
        return tensor.get_shape().as_list()
    else:
        static_shape = tensor.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(tensor), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def ssd_multibox_layers(
        inputs,
        num_classes,
        sizes,
        ratios=[1],
        normalization=-1,
        bn_normalization=False
):
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)

    '''
    Generate correspondingly 4,6,6,6,6,4 default boxes for each feature point
    on layer 4,7,8,9,10,11,12
    why len(sizes) + len(ratios)? not len(sizes) * len(ratios)?
    square: min_size, (min_size*max_size)^0.5                       2,2,2,2,2,2
                    +                                                    +
    rectangle: ratio^0.5 * min_size, 1/ration^0.5 *min_size         2,4,4,4,4,2
                    =                                                    =
                                                                    4,6,6,6,6,4  
    '''
    num_anchors = len(sizes) + len(ratios)
    #Location
    num_loc_pred = num_anchors * 4          #4 coordinates for each anchor
    loc_pred = slim.conv2d(net,             #Output is [H of feature map, W of feature map, anchor_num * 4]
                           num_loc_pred,
                           [3, 3],
                           activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(                                  #E.g. The output for layer4 will be:
        loc_pred,                                           #[0, 64, 64, num_anchors(4), 4]
        tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4]
    )
    #Class
    num_cls_pre = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pre, [3, 3], activation_fn=None, scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(
        cls_pred, tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes]
    )
    return cls_pred, loc_pred

def ssd_net(
        inputs,
        num_classes,
        feature_layers,
        anchor_sizes,
        anchor_ratios,
        normalizations,
        is_training=True,
        dropout_keep_prob=0.5,
        prediction_fn=slim.softmax,
        reuse=None,
        scope='ssd_vgg'
):
    #structure of SSD net

    outputs = {}
    with tf.variable_scope(scope, 'ssd_vgg', [inputs], reuse=reuse):
        #Structure of vgg16
        #Block1
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        outputs['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        #Block 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        outputs['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #Block 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        outputs['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        #Block 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        outputs['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #Block 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        outputs['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        #Additional SSD blocks
        #Block 6
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        outputs['block6'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        #block 7
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        outputs['block7'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        #Block8
        with tf.variable_scope('block8'):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1,1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        outputs['block8'] = net
        #Block 9
        with tf.variable_scope('block9'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1,1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        outputs['block9'] = net
        # Block 10
        with tf.variable_scope('block10'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        outputs['block10'] = net
        # Block 11
        with tf.variable_scope('block11'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        outputs['block11'] = net
        # Block 12
        with tf.variable_scope('block12'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [4, 4], stride=2, scope='conv4x4', padding='VALID')
        outputs['block12'] = net

        #Prediction and locolization
        predictions = []                               #Class prediction
        logits = []                                    #Probability of class
        locations = []                             #Location prediction
        for i, layer in enumerate(feature_layers):            #Block 4,7,8,9,10,11,12
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(
                    outputs[layer],
                    num_classes,
                    anchor_sizes[i],
                    anchor_ratios[i],
                    normalizations[i]
                )
                predictions.append(prediction_fn(p))   #Here use softmax to predict classs
            logits.append(p)
            locations.append(l)
    return predictions, locations, logits, outputs


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [0, 512, 512, 3])
    net = ssd_net(inputs, 1, 1, 1, 1, None)
    print(net)
    for key, value in net.items():
        print(key,':', value.shape)
    # with tf.Session() as sess:
        # summary_writer = tf.summary.FileWriter('../log/', sess.graph)

    print('OK')
