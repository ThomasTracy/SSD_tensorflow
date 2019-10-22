from datasets import voc07

datasets_map = {
    pascalvoc_2007: voc07
}

def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):

    if name not in datasets_map:
        raise ValueError('This dataset: %s does not exist' % name)
    return datasets_map[name].get_split(split_name,dataset_dir, file_pattern, reader)