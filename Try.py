from PIL import Image
from matplotlib import pyplot as plt
import numpy
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import custom_layers, ssd_vgg

def anchor_points_show(x, y):
    plt.scatter(y, x, c='b', marker='.')
    plt.grid(True)
    plt.show()

def anchor_one_layer_show(image, x, y, w, h):
    img = Image.open(image)
    image_size = img.size
    x = x.flatten()
    y = y.flatten()
    x_expand = x * image_size[0]
    y_expand = y * image_size[1]
    w_expand = w * image_size[0]
    h_expand = h * image_size[1]
    plt.figure('Image')
    plt.imshow(img)
    # for (x_draw, y_draw) in zip(x_expand, y_expand):
    #     for (w_draw, h_draw) in zip(w_expand, h_expand):
    #         plt.gca().add_patch(plt.Rectangle((x_draw, y_draw), w_draw, h_draw, edgecolor='r', linewidth=1))
    for (w_draw, h_draw) in zip(w_expand, h_expand):
        x_draw = 500. - w_draw/2
        y_draw = 500. - h_draw/2
        plt.gca().add_patch(plt.Rectangle((x_draw, y_draw), w_draw, h_draw, edgecolor='r', linewidth=1, facecolor='None'))
    plt.show()

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

anchor_one_layer_show(image, x, y, w, h)
print()



