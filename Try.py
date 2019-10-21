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

image = "D:\\Pycharm\\Projects\\SSD_tensorflow\\test.jpg"
y, x, h, w = ssd_vgg.ssd_anchor_one_layer(
                                img_shape=(512, 512),
                                layer_shape=(64, 64),
                                anchor_size=(296.96, 378.88),
                                anchor_ratio=(2, .5, 3, 1./3),
                                anchor_step=128,
                                offset=0.5,
                                dtype=numpy.float32
                                )

i = 0
n =10
def cond(i, n):
    return i < n

def body(i, n):
    i = i + 1
    print(i)
    print("hahahahaha")
    return i, n
# i, n = tf.while_loop(cond, body, [i, n])

a = tf.ones(shape=[3,3])
b = tf.reshape(tf.range(0,24, dtype=tf.float32), [2,3,4])
x = tf.reshape(b, [-1])
value, index = tf.nn.top_k(x, k=3)

c = tf.argmax(b, axis=2)
d = tf.reduce_max(b, axis=2)
prediction = slim.softmax(b)

logit = tf.random_normal([5,20], mean=1)
label = numpy.random.randint(2, size=(5,20))

new_b = tf.transpose(b, perm=(2, 0, 1))
print(new_b.get_shape())

def apply_with_randon_selector(x, func, num_case):
    sel = tf.random_uniform([], maxval=num_case, dtype=tf.int32)
    pass

def show(x=0, case=0):
    # print("Now the selected is: {} \nAnd case is: {} ".format(x, case))
    return x

sel = numpy.random.randint(5)
print(sel)
result = control_flow_ops.merge([show(control_flow_ops.switch(
          sel, tf.equal(sel, case))[1], case)
          for case in range(5)])

# with tf.Session() as sess:
    # print(sess.run(result))