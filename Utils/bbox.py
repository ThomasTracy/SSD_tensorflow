import numpy
from Utils import tensor
import tensorflow as tf

def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    '''

    :param scores: (batch, 64x64x4)
    :param bboxes: (batch, 64x64x4, 4)
    :param top_k:
    :param scope:
    :return:
    '''
    #Dictionary Input
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)   #Reuse self. This time with Tensor input
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    #Tensor Input
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        scores, index = tf.nn.top_k(scores, k=top_k, sorted=True)

        def fn_gather(bboxes, index):
            bb = tf.gather(bboxes, index)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, index],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes

def bboxes_clip(bbox_ref, bboxes, scope=None):      #Clipping bbox to intersected area with reference bbox
    #Dictionary
    if isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_clip_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_clip(bbox_ref, bboxes[c])
            return d_bbxes
    #Tensor
    with tf.name_scope(scope, 'bboxes_clip'):
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)

        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])

        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes

def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):

    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):

        index = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores,index)
        bboxes = tf.gather(bboxes, index)
        scores = tensor.pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = tensor.pad_axis(scores, 0, keep_top_k, axis=0)
        return scores, bboxes

def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.
    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes