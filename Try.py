from PIL import Image
from tools import anchors_show
from matplotlib import pyplot as plt
import numpy
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import custom_layers, ssd_vgg

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

c = tf.argmax(b, axis=2)
d = tf.reduce_max(b, axis=2)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))