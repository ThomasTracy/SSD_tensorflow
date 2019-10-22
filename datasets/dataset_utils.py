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


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.
    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.
    Returns:
    A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read()
    lines = lines.split(b'\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(b':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names