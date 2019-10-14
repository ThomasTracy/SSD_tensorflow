import numpy
import tensorflow as tf

def ssd_bbox_encode_one_layer(labels,
                              bboxes,                   #(N x 4) of GT bounding boxes
                              anchor_layer,
                              num_classes,
                              no_annotation_label,
                              gt_threshold=0.5,
                              prior_scaling=[0.1, 0.1, 0.2, 0.2],
                              dtype = tf.float32):
    '''
    Encode ground truth labels and bounding boxes
    Returned Tensor: (target label, target location, target score)
    '''
    #Coordinate and size of anchors
    y_ref, x_ref, h_ref, w_ref = anchor_layer               #From proposed anchors
    y_min = y_ref - h_ref/2                                 #top left point and bottom right point
    x_min = x_ref - w_ref/2                                 #e.g. Block 4: x_ref, y_ref (64, 64 ,1), w, h (4, )
    y_max = y_ref + h_ref/2                                 #y_min: (64, 64, 1) - (4,) = (64, 64, 4)!!!!!!!
    x_max = x_ref + w_ref/2
    anchor_area = (y_max - y_min) * (x_max - x_min)

    shape = (y_ref.shape[0], y_ref.shape[1], h_ref.size)    #Block4 (64, 64, 4) Block7 (30, 30, 6)
    feat_labels = tf.zeros(shape=shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape=shape, dtype=tf.float32)

    feat_y_min = tf.zeros(shape=shape, dtype=dtype)         #每个像素点有四种大小的bbox，因此为（64，64，4）维
    feat_x_min = tf.zeros(shape=shape, dtype=dtype)         #或者为（N, N，6）
    feat_y_max = tf.ones(shape=shape, dtype=dtype)
    feat_x_max = tf.ones(shape=shape, dtype=dtype)

    def jaccard_with_anchor(bbox):
        #Overlapping percentage (IoU) between GT bbox and proposed bbox A∩B / A∪B
        overlap_ymin = tf.maximum(y_min, bbox[0])   #Overlap_ymin: (64, 64, 1) bbox[0] is a number
        overlap_xmin = tf.maximum(x_min, bbox[1])   #Is bbox[0] a number? Or (batch, 64, 64, 4) ????
        overlap_ymax = tf.maximum(y_max, bbox[2])
        overlap_xmax = tf.maximum(x_max, bbox[3])

        h = tf.maximum(overlap_ymax - overlap_ymin, 0.)
        w = tf.maximum(overlap_xmax - overlap_xmin, 0.)
        overlap_area = w * h
        union_area = anchor_area - overlap_area + \
                     (bbox[2] - bbox[0])*(bbox[3] - bbox[1])           # A∪B = A - A∩B + B
        jaccard = tf.div(overlap_area, union_area)                     #Jaccard: (64, 64, 4)!!!!
        return jaccard

    def intersection_with_anchor(bbox):                 #A∩B / B  score of proposed anchor
        overlap_ymin = tf.maximum(y_min, bbox[0])
        overlap_xmin = tf.maximum(x_min, bbox[1])
        overlap_ymax = tf.maximum(y_max, bbox[2])
        overlap_xmax = tf.maximum(x_max, bbox[3])

        h = tf.maximum(overlap_ymax - overlap_ymin, 0.)
        w = tf.maximum(overlap_xmax - overlap_xmin, 0.)
        overlap_area = w * h
        scores = tf.div(overlap_area, anchor_area)
        return scores

    def cond(i, feat_labels, feat_scores, feat_y_min, feat_y_max, feat_x_min, feat_x_max):
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores, feat_y_min, feat_y_max, feat_x_min, feat_x_max):
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchor(bbox)                 #(64, 64, 4)
        mask = tf.greater(jaccard, feat_scores)             #(64, 64, 4)
        #mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)    #In order to garentee label <21
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)

        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_y_min = fmask * bbox[0] + (1 - fmask) * feat_y_min
        feat_y_max = fmask * bbox[1] + (1 - fmask) * feat_y_max
        feat_x_min = fmask * bbox[2] + (1 - fmask) * feat_x_min
        feat_x_max = fmask * bbox[3] + (1 - fmask) * feat_x_max

        # inter = intersection_with_anchor(bbox)
        # mask = tf.logical_and(inter > gt_threshold, label == no_annotation_label)   #Ignore no annotated labels
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)
        return [i + 1, feat_labels, feat_scores, feat_y_min, feat_y_max, feat_x_min, feat_x_max]

    i = 0
    [i, feat_labels, feat_scores, feat_y_min, feat_y_max, feat_x_min, feat_x_max] = \
    tf.while_loop(cond, body,
                  [i, feat_labels, feat_scores, feat_y_min, feat_y_max, feat_x_min, feat_x_max])

    #From top left, bottom right to center, width, height
    feat_cy = (feat_y_min + feat_y_max) / 2
    feat_cx = (feat_x_min + feat_x_max) / 2
    feat_h = feat_y_max - feat_y_min
    feat_w = feat_x_max - feat_x_min

    '''
    Returned is not simply location, but the Tranformation
    between proposed bbox and GT bbox
    '''
    feat_cy = (feat_cy - y_ref) / y_ref / prior_scaling[0]      #(64, 64, 4)?
    feat_cx = (feat_cx - x_ref) / x_ref / prior_scaling[1]
    feat_h = tf.log(feat_h / h_ref) / prior_scaling[2]          #Also (64, 64, 4)?
    feat_w = tf.log(feat_w / h_ref) / prior_scaling[3]

    feat_locations = tf.stack([feat_cy, feat_cx, feat_h, feat_w], axis=-1)      #(64, 64, 4, 3)
    # 这个地方损失函数其实是我们预测的是变换，我们实际的框和anchor之间的变换和我们预测的变换之间的loss。
    # 我们回归的是一种变换。并不是直接预测框，这个和YOLO是不一样的。和Faster RCNN是一样的
    return feat_labels, feat_locations, feat_scores

def ssd_bboxes_encode(labels,           #From absolut boxes to relative Translation
                     bboxes,
                     anchors,
                     num_classes,
                     no_annotation_label,
                     gt_threshold=0.5,
                     prior_scaling=[0.1, 0.1, 0.2, 0.2],
                     dtype=tf.float32,
                     scope='ssd_bboxed_encode'):
    with tf.name_scope(scope):
        target_labels = []
        target_locations = []
        target_scores = []
        for i, anchor_layer in enumerate(anchors):
            with tf.name_scope('bbox_encode_block_%i' %i):
                t_label, t_loc, t_score = \
                ssd_bbox_encode_one_layer(labels, bboxes,
                                          anchor_layer, num_classes,
                                          no_annotation_label,
                                          gt_threshold, prior_scaling, dtype)
            target_labels.append(t_label)
            target_locations.append(t_loc)
            target_scores.append(t_score)
        return target_labels, target_locations, target_scores

def ssd_bboxes_decode_layer(feat_location,              #(64, 64 ,4)
                            anchors_layer,
                            prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    y_ref, x_ref, h_ref, w_ref = anchors_layer

    cx = feat_location[:, :, :, :, 0] * w_ref * prior_scaling[0] + x_ref    #Dimension of feat_loc is 5???????
    cy = feat_location[:, :, :, :, 1] * h_ref * prior_scaling[1] + y_ref    #(batch, 64, 64, 4, stack(4))??????
    w = w_ref * tf.exp(feat_location[:, :, :, :, 2] * prior_scaling[2])
    h = h_ref * tf.exp(feat_location[:, :, :, :, 3] * prior_scaling[3])

    y_min = cy - h/2
    y_max = cy + h/2
    x_min = cx - w/2
    x_max = cx + w/2

    bboxes = tf.stack([y_min, x_min, y_max, x_max], axis=-1)
    return bboxes

def ssd_bboxes_decode(feat_locations,
                      anchors,
                      prior_scaling=[0.1, 0.1, 0.2, 0.2],
                      scope='ssd_bboxes_decode'):
    with tf.name_scope(scope):
        bboxes = []
        for i, anchor_layer in enumerate(anchors):
            bboxes.append(ssd_bboxes_decode_layer(feat_locations,
                                                  anchor_layer,
                                                  prior_scaling))
        return bboxes

#-----------------------------------------------------
#                    Boxes Selection
#-----------------------------------------------------
def ssd_bboxes_select_layer(predictions_layer, locations_layer,
                            select_threshold=None,
                            num_classes=21,
                            ignore_class=0,
                            scope=None):
    if select_threshold is None:
        select_threshold = 0.0
    else:
        select_threshold = select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, locations_layer]):
        p_shape = predictions_layer.get_shape()                     #Not the same with original Code!!!!!!!!!!
        predictions_layer = tf.reshape(predictions_layer,           #From (1, 64, 64, 4, 21) to (1, 64x64x4, 21)
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = locations_layer.get_shape()
        locations_layer = tf.reshape(locations_layer,               #From (1, 64, 64, 4, 4) to (1, 64x64x4, 4)
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))

        dic_score = {}
        dic_bbox = {}

        for i_class in range(0, num_classes):
            if i_class != ignore_class:
                score = predictions_layer[:, :, i_class]                    #(1, 64x64x4)
                fmask = tf.cast(tf.greater_equal(score, select_threshold), score.dtype)
                score = score * fmask                           #(1, 64x64x4)
                bboxes = locations_layer * tf.expand_dims(fmask, dim=-1)    #(1, 64x64x4, 4)

                dic_score[i_class] = score          #(1, 64x64x4)
                dic_bbox[i_class] = bboxes      #(1, 64x64x4, 4)

        return dic_score, dic_bbox

def ssd_bboxes_select(predictions_net, locations_net,
                      select_threshold=None,
                      num_classes=21,
                      ignore_class=0,
                      scope=None):
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_layer, locations_layer]):
        l_score = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = ssd_bboxes_select_layer(predictions_net[i],
                                                     locations_net[i],
                                                     select_threshold,
                                                     num_classes,
                                                     ignore_class)
            l_score.append(scores)                      #N x (1, 64x64x4)
            l_bboxes.append(bboxes)                     #N x (1, 64x64x4, 4)

        dic_score = {}
        dic_bboxes = {}
        for c in l_score[0].keys():                     #A 21 classes
            ls = [s[c] for s in l_score]                #All class x from list of dictionary
            lb = [b[c] for b in l_bboxes]
            dic_score[c] = tf.concat(ls, axis=1)        #(1, 64x64x4xN)
            dic_bboxes[c] = tf.concat(lb, axis=1)       #(1, 64x64x4xN, 4)
        return dic_score, dic_bboxes

def ssd_bboxes_select_layer_all_classes(predictions_layer, locations_layer,
                                        select_threshold=None):
    p_shape = predictions_layer.get_shape()
    predictions_layer = tf.reshape(predictions_layer,               #From (1, 64, 64, 4, 21) to (1, 64x64x4, 21)
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = locations_layer.get_shape()
    locations_layer = tf.reshape(locations_layer,                   #From (1, 64, 64, 4, 4) to (1, 64x64x4, 4)
                                 tf.stack([l_shape[0], -1, l_shape[1]]))

    if select_threshold is None or select_threshold == 0:
        classes = tf.argmax(predictions_layer, axis=2)
        scores = tf.reduce_max(predictions_layer, axis=2)
        scores = scores * tf.cast(classes > 0, scores.dtype)
    else:
        '''
        prediction_layer: [[[0,1,2...21], [0,1,2...21]...[0,1,2...21]]]     (1, 64x64x4, 21)
        classes: [[class_id, class_id...class_id]]      (1, 64x64x4)
        scores: [[max_score, max_score, ...max_score]]  (1, 64x64x4)
        '''
        sub_pre = predictions_layer[:, :, 1:]               #Without background
        classes = tf.argmax(sub_pre, axis=2) + 1
        scores = tf.reduce_max(sub_pre, axis=2)
        mask = tf.greater(scores, select_threshold)
        classes = classes * tf.cast(mask, classes.dtype)
        scores = scores * tf.cast(mask, scores.dtype)
    bboxes = locations_layer
    return classes, scores, bboxes

def ssd_bboxes_select_all_classes(predictions_net, locations_net,
                                  select_threshold=None,
                                  scope=None):
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, locations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = \
                ssd_bboxes_select_layer_all_classes(predictions_net[i],
                                                    locations_net[i],
                                                    select_threshold)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)
        return classes, scores, bboxes
