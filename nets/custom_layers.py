import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope


@add_arg_scope
def l2_normalization(
        inputs,
        scaling=False,                                          #Scaling after normalization
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        data_format='NHWC',
        trainable=True,
        scope=None
):
    with variable_scope.variable_scope(
        scope,
        'L2Normalization',
        [inputs],
        reuse=reuse
    ) as sc:
        inputs_shape = inputs.get_shape()       #[N, H, W, C]
        inputs_rank = inputs_shape.ndims        #dimension 4
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            norm_dim = tf.range(inputs_rank-1, inputs_rank)     #Choose dimension 'C' from 'NHWC'
            params_shape = inputs_shape[-1:]                    #How many channels
        elif data_format == 'NCHW':
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)       #Normalizing

        if scaling:
            scale_collections = utils.get_variable_collections(variables_collections, 'scale')
            scale = variables.model_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=scale_initializer,
                collections=scale_collections,
                trainable=trainable
            )
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs)

        return utils.collect_named_outputs(outputs_collections, sc.original_name_scope, outputs)

@add_arg_scope
def pad2d(
        inputs,
        pad=(0,0),
        mode='CONSTANT',
        data_format='NHWC',
        trainable=True,
        scope=None
):
    with tf.name_scope(scope, 'pad2d', [inputs]):
        if data_format == 'NHWC':
            paddings = [[0,0], [pad[0],pad[0]], [pad[1], pad[1]], [0,0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0] ,[pad[0], pad[0]], [pad[1], pad[1]]]
        out = tf.pad(inputs, paddings, mode=mode)
        return out

@add_arg_scope
def channel_to_last(            #Push the channel dimension to the last position
    inputs,
    data_format='NHWC',
    scope=None
):
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            out = inputs
        elif data_format == 'NCHW':
            out = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return out


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r