import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pprint import pprint
from tensorflow.contrib.slim.python.slim.data import parallel_reader


def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.
    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


def print_configs(flags, ssd_params, data_sources, save_dir=None):

    def print_cnf(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# SSD net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(dict(ssd_params._asdict()), stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    print_cnf(None)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            path = os.path.join(save_dir, 'training_config.txt')
            with open(path, 'w') as f:
                print_cnf(f)