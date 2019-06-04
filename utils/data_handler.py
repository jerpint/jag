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

from tokenizer.bert_tokenization import BertTokenizer


class DatasetHandler():
    """ Class with different tools for the dataset.

    """

    def __init__(self, data_src: str = 'mrqa_urls.txt', cache_dir: str = '.'):
        self.dpath_dict = self.get_file(data_src, cache_dir)
        print(self.dpath_dict)

    def preprocess(self, s: str) -> str:
        """Preprocess the string s. These operations are from the evaluation script
        of the MRQA challenge."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def remove_html_tags(text):
            return re.sub(r'<(.|\n)*?>', ' ', text)

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
                splits = line.split(' ')
                if len(splits) == 3:
                    mode, url, file_hash = splits
                    fname = os.path.basename(url)
                elif len(splits) == 4:
                    mode, fname, url, file_hash = splits
                else:
                    raise ValueError("unkonwn format")
                cache_subdir = 'datasets/' + mode
                dpath = tf.keras.utils.get_file(
                    fname=fname,
                    origin=url,
                    file_hash=file_hash,
                    cache_subdir=cache_subdir,
                    cache_dir=cache_dir)
                dpath_dict[fname.split('.')[0] + "_" + mode] = dpath
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
        raw_file = open(rawdata_path, 'w')
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
                    raw_file.write('{}\n'.format(
                        self.preprocess(qa['question'])))
                    for ans in qa['detected_answers']:
                        raw_file.write('{}\n'.format(
                            self.preprocess(ans['text'])))
        print('Processing completed. Writing raw data to {}'.format(rawdata_path))
        raw_file.close()

    @staticmethod
    def reconstruct_text_from_tokens(tokens, wordpiece_indicator="##") -> str:
        """ Reconstruct a text from a list of tokens given a wordpiece indicator on tokens.
        Args:
            tokens (List[str]): the list of tokens
            wordpiece_indicator (str): an indicator of a subword tokens
        Returns:
            text: the reconstructed text
        """
        n = 0 if wordpiece_indicator is None else len(wordpiece_indicator)
        result = []
        i = 0
        while i < len(tokens):
            if (
                (n == 0) or
                (len(tokens[i]) < n) or
                (len(tokens[i]) >= n and tokens[i][:n] != wordpiece_indicator)
            ):
                result.append(tokens[i])
                i = i + 1
            else:
                data = result[-1]
                j = i
                while (
                    (j < len(tokens)) and
                    (len(tokens[j]) >= n and tokens[
                     j][:n] == wordpiece_indicator)
                ):
                    data += tokens[j][n:]
                    j = j + 1

                result[-1] = data
                i = j

        return " ".join(result)

    def check_answer_tokenization_from_dataset(
            self, dataset_name, tokenizer, wordpiece_indicator="##"):
        """ Check the proportion of the answers in the given dataset that can be
        perfectly reconstructed using the provided tokenizer
        Args:
            dataset_path (str): path of the dataset
            tokenizer: the tokenizer to be used
            wordpiece_indicator (str): an indicator of a subword tokens
        Returns:
            result (float, int): reconstruction rate and number of
            examples of the dataset
        """
        f = gzip.open(self.dpath_dict[dataset_name], 'rb')
        num_examples = 0
        well_reconstructed = 0
        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in ex:
                continue

            for qa in ex['qas']:
                for v in qa['answers']:
                    num_examples += 1

                    text = self.preprocess(v)
                    tokens = tokenizer.tokenize(text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    reconstructed_tokens = tokenizer.convert_ids_to_tokens(
                        token_ids)
                    reconstructed_text = DatasetHandler.reconstruct_text_from_tokens(
                        reconstructed_tokens
                    )

                    if reconstructed_text == text:
                        well_reconstructed += 1
        return (well_reconstructed * 100.0) / max(1, num_examples), num_examples

    def check_answer_tokenization(
            self, tokenizer, wordpiece_indicator="##") -> Dict:
        """ Check the proportion of the answers in each dataset that can be
        perfectly reconstructed using the provided tokenizer
        Args:
            tokenizer: the tokenizer to be used
            wordpiece_indicator (str): an indicator of a subword tokens
        Returns:
            result (Dict[str, (float, int)]): the reconstruction rate for each dataset
        """
        result = {}
        for dataset_name in self.dpath_dict.keys():
            print(
                'Processing the reconstruction rate of dataset: {}'.format(
                    dataset_name)
            )
            rate = self.check_answer_tokenization_from_dataset(
                dataset_name, tokenizer, wordpiece_indicator)
            result[dataset_name] = rate
            print('Results Obtained so far: {}'.format(result))
        return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='./')
    parser.add_argument('--raw_data_path', type=str, default=None)
    args = parser.parse_args()

    data_handler = DatasetHandler(cache_dir=args.cache_dir)

    tokenizer = BertTokenizer.from_pretrained(
        # 'bert-base-uncased', 'bert-base-multilingual-uncased'
        pretrained_model_name_or_path='bert-base-multilingual-cased',
        cache_dir=args.cache_dir,
        do_lower_case=True,
        max_len=1e12,
        do_basic_tokenize=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]",
                     "[MASK]", "[TLE]", "[DOC]", "[PAR]")
    )

    # vocab = data_handler.build_vocab()
    # print(len(vocab))
    data_handler.generate_raw_dataset()
