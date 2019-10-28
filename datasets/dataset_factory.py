from datasets import voc07


FILE_PATTERN = 'voc_%s_*.tfrecord'
datasets_map = {
    'pascalvoc_2007': voc07
}

def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):

    if not file_pattern:
        file_pattern = FILE_PATTERN
    if name not in datasets_map:
        raise ValueError('This dataset: %s does not exist' % name)
    return datasets_map[name].get_split(split_name, dataset_dir, file_pattern, reader)