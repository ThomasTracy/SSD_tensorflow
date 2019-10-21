import os
import tensorflow as tf


LABELS_FILENAME = 'labels.txt'


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.
    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.
    Returns:
    `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))