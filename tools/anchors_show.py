from matplotlib import pyplot as plt
import numpy
from PIL import Image


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