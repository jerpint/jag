# -*- coding: utf-8 -*-
"""
Data handling utils.

"""
import argparse
import gzip
import json
import os
import re
import string
import tensorflow as tf
from typing import Dict, List


class DatasetHandler():
    """ Class with different tools for the dataset.

    """
    def __init__(self, data_src: str = 'train_urls.txt', cache_dir: str = '.'):
        self.dpath_dict = self.get_file(data_src, cache_dir)
        print(self.dpath_dict)

    def preprocess(self, s: str) -> str:
        """Preprocess the string s. These operations are from the evaluation script
        of the MRQA challenge."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

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

    def build_vocab(self, vocab_path: str = 'vocab.txt') -> List[str]:
        """ Retrieve the union of vocabularies from different datasets.

        Args:
            vocab_path (str): the filepath of the vocabulary file

        Returns:
            vocab: A list of tokens

        """
        vocab = []
        for dataset_name in self.dpath_dict.keys():
            print('Processing the vocabulary of dataset: {}'.format(dataset_name))
            f = gzip.open(self.dpath_dict[dataset_name], 'rb')
            for i, line in enumerate(f):
                ex = json.loads(line)

                # Skip headers.
                if i == 0 and 'header' in ex:
                    continue

                vocab += [l[0].lower() for l in ex['context_tokens']]
                for qa in ex['qas']:
                    vocab += [l[0].lower() for l in qa['question_tokens']]
            vocab = list(set(vocab))
            print('Vocabulary length so far: {:d}'.format(len(vocab)))
        print('Processing completed. Writing vocabulary to {}'.format(vocab_path))
        vocab.sort()
        with open(vocab_path, 'w') as vfile:
            for token in vocab:
                vfile.write('{}\n'.format(token))
        return vocab

    def generate_raw_dataset(self, rawdata_path: str = 'rawdata.txt'):
        """ Generate a raw dataset in a text file where each line is either
        a context, a question or an answer.

        Args:
            rawdata_path (str): path of the new raw data text file

        Returns:
            None

        """
        raw_file = open(rawdata_path, 'w', encoding='utf-8')
        for dataset_name in self.dpath_dict.keys():
            print('Processing the dataset: {}'.format(dataset_name))
            f = gzip.open(self.dpath_dict[dataset_name], 'rb')
            for i, line in enumerate(f):
                ex = json.loads(line)

                # Skip headers.
                if i == 0 and 'header' in ex:
                    continue

                raw_file.write(self.preprocess(ex['context']))
                for qa in ex['qas']:
                    raw_file.write('{}\n'.format(self.preprocess(qa['question'])))
                    for ans in qa['detected_answers']:
                        raw_file.write('{}\n'.format(self.preprocess(ans['text'])))
        print('Processing completed. Writing raw data to {}'.format(rawdata_path))
        raw_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--raw_data_path', type=str, default=None)
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--raw_data_path', type=str, default=None)
    args = parser.parse_args()

    data_handler = DatasetHandler(cache_dir=args.cache_dir)
    if args.vocab_path is not None:
        vocab = data_handler.build_vocab()
        print(len(vocab))
    if args.raw_data_path is not None:
        data_handler.generate_raw_dataset(args.raw_data_path)
