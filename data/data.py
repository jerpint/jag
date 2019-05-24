# -*- coding: utf-8 -*-
"""
Data handling utils.

"""

import gzip
import json
import os
import tensorflow as tf
from typing import Dict, List


class DatasetHandler():
    """ Class with different tools for the dataset.

    """
    def __init__(self, data_src='mrqa_urls.txt', cache_dir=None):
        self.dpath_dict = self.get_file(data_src, cache_dir)
        print(self.dpath_dict)

    def get_file(self, data_src: str, cache_dir: str) -> Dict[str, str]:
        """ Retrieve the path of the datasets specified in data_src in cache_dir.
        If the dataset does not exists in cache_dir, it is downloaded.

        Args:
            data_src: The path of the file with the sources of the datasets
            cache_dir: The path of the cache directory

        Returns:
            dpath_dict: A dictionary with pairs (dataset_name: path)

        """
        dpath_dict = {}
        for line in open(data_src, 'r').read().splitlines():
            if line[0] != '#':
                url, file_hash = line.split(' ')
                dpath = tf.keras.utils.get_file(
                        fname=os.path.basename(url),
                        origin=url,
                        file_hash=file_hash,
                        cache_dir=cache_dir)
                dpath_dict[os.path.basename(url).split('.')[0]] = dpath
        return dpath_dict

    def build_vocab_from_dataset(self, dataset_name: str) -> List[str]:
        """ Retrieve the path of the datasets specified in data_src in cache_dir.
        If the dataset does not exists in cache_dir, it is downloaded.

        Args:
            dataset_path (str): path of the dataset

        Returns:
            vocab: A list of tokens

        """
        f = gzip.open(self.dpath_dict[dataset_name], 'rb')
        vocab = []
        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in ex:
                continue

            vocab += [l[0].lower() for l in ex['context_tokens']]
            for qa in ex['qas']:
                vocab += [l[0].lower() for l in qa['question_tokens']]
        return vocab

    def build_vocab(self, vocab_filepath: str = 'vocab.txt') -> List[str]:
        """ Retrieve the union of vocabularies from different datasets.

        Args:
            vocab_filepath (str): the filepath of the vocabulary file

        Returns:
            vocab: A list of tokens

        """
        vocab = []
        for dataset_name in self.dpath_dict.keys():
            print('Processing the vocabulary of dataset: {}'.format(dataset_name))
            vocab += self.build_vocab_from_dataset(dataset_name)
            vocab = list(set(vocab))
            print('Vocabulary length so far: {:d}'.format(len(vocab)))
        print('Processing completed. Writing vocabulary to {}'.format(vocab_filepath))
        vocab.sort()
        with open(vocab_filepath, 'w') as vfile:
            for token in vocab:
                vfile.write('{}\n'.format(token))
        return vocab


if __name__ == '__main__':
    data_handler = DatasetHandler()
    vocab = data_handler.build_vocab()
    print(len(vocab))
