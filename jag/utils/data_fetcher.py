import tensorflow as tf
import os
from typing import Dict


def get_file(data_src: str, cache_dir: str) -> Dict[str, str]:
    """ Return the local path of the datasets specified in data_src in cache_dir.
    If a dataset does not exists in cache_dir or its MD5 does not agree, it is downloaded.
    Args:
        data_src: path of the file with the sources of the datasets
        cache_dir: path of the cache directory
    Returns:
        dpath_dict: A dictionary with pairs (dataset_name: path)
    """

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    dpath_dict = {}
    for line in open(data_src, 'r').read().splitlines():
        if line[0] != '#':
            splits = line.split(' ')
            if len(splits) == 3:
                mode, url, file_hash = splits
                fname = os.path.basename(url)
            elif len(splits) == 4:
                mode, fname, url, file_hash = splits
            else:
                raise ValueError(
                    "unknown format: the format must be 'mode url md5' per line")
            cache_subdir = 'datasets/' + mode
            dpath = tf.keras.utils.get_file(
                fname=fname,
                origin=url,
                file_hash=file_hash,
                cache_subdir=cache_subdir,
                cache_dir=cache_dir)
            dpath_dict[fname.split('.')[0] + "_" + mode] = dpath
    return dpath_dict
