# -*- coding: utf-8 -*-
"""
Data handling utils.
"""

import gzip
import json
import os
import tensorflow as tf
from typing import Dict, List
import pickle
import string
import re
import sys
from collections import Counter

from tokenization import BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP


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

    def normalize_text(self, text) -> str:
        """ Lower text and remove punctuation, articles and extra whitespace.
        Args:
            text (str): text to normalize
        Returns:
            normalized_text: normalized text
        """
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

        return white_space_fix(
            remove_articles(
                remove_punc(
                    remove_html_tags(
                        # lower(
                        text
                        #)
                    )
                )
            )
        )

    def retrieve_max_contexlen_from_dataset(self, dataset_name, tokenizer) -> int:
        """ Retrieve the path of the datasets specified in data_src in cache_dir.
        If the dataset does not exists in cache_dir, it is downloaded.
        Args:
            dataset_path (str): path of the dataset
        Returns:
            extracted_dataset: A list of tuple containing the raw normalized context,
            the related normalized questions and the associated normalized answers.
        """
        f = gzip.open(self.dpath_dict[dataset_name], 'rb')
        maxval = 0
        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in ex:
                continue

            context = self.normalize_text(ex['context'])
            tokens = tokenizer.tokenize(context)

            if len(tokens) > 800:
                print('raw context: \n', ex['context'])
                print('tokens: \n', tokens, len(tokens))
                print('context_tokens: \n', ex[
                      'context_tokens'], len(ex['context_tokens']))
                print('raw context: \n', ex['context'])
                reconstructed_text = DatasetHandler.reconstruct_text_from_tokens(
                    tokens
                )
                print('raw context: \n', reconstructed_text)
                exit(0)

            maxval = max(maxval, len(context))

        return maxval

    def retrieve_max_contexlen(self, tokenizer) -> Dict:
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
            rate = self.retrieve_max_contexlen_from_dataset(
                dataset_name, tokenizer)
            result[dataset_name] = rate
            print('Results Obtained so far: {}'.format(result))
        return result

    def extract_normalized_text_from_dataset(self, dataset_name: str) -> List:
        """ Retrieve the path of the datasets specified in data_src in cache_dir.
        If the dataset does not exists in cache_dir, it is downloaded.
        Args:
            dataset_path (str): path of the dataset
        Returns:
            extracted_dataset: A list of tuple containing the raw normalized context,
            the related normalized questions and the associated normalized answers.
        """
        f = gzip.open(self.dpath_dict[dataset_name], 'rb')
        extracted_dataset = []
        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in ex:
                continue

            context = self.normalize_text(ex['context'])
            question_answers = []

            for qa in ex['qas']:
                question_answers.append([
                    qa['qid'],
                    self.normalize_text(qa['question']),
                    [self.normalize_text(v) for v in qa['answers']]
                ])
            extracted_dataset.append([dataset_name, context, question_answers])
        return extracted_dataset

    def extract_normalized_text(
            self, normalized_data_filepath: str = 'normalized_data.txt') -> List[str]:
        """ Retrieve the union of normalized text from different datasets.
        Args:
            normalized_data_filepath (str): the filepath of the normalized text file
        Returns:
            normalized_data: A list of normalized text
        """
        normalized_data = []
        for dataset_name in self.dpath_dict.keys():
            print('Processing the normalized text of dataset: {}'.format(dataset_name))
            normalized = self.extract_normalized_text_from_dataset(
                dataset_name)
            pickle.dump(
                normalized,
                open("{}.normalized.p".format(dataset_name), "wb")
            )
            for d in normalized:
                normalized_data.append(d[1])
                for a in d[2]:
                    normalized_data.append(a[1])
                    normalized_data.extend(a[2])

            print('Normalized text length so far: {:d}'.format(
                len(normalized_data)))
        print('Processing completed. Writing vocabulary to {}'.format(
            normalized_data_filepath
        ))

        with open(normalized_data_filepath, 'w') as vfile:
            for token in normalized_data:
                vfile.write('{}\n'.format(token))
        return normalized_data

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

    @staticmethod
    def reconstruct_text_from_tokens(tokens, wordpiece_indicator="##") -> str:
        """ Reconstruct a text from a list of tokens given a wordpiece indicator on tokens.
        Args:
            tokens (List[str]): the list of tokens
            wordpiece_indicator (str): an indicator of a subword tokens
        Returns:
            text: the reconstructed text
        """
        n = len(wordpiece_indicator)
        result = []
        i = 0
        while i < len(tokens):
            if (
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

                    text = self.normalize_text(v)
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

    data_handler = DatasetHandler(
        data_src='../mrqa_train_urls.txt', cache_dir='../'
    )

    # normalized_data = data_handler.extract_normalized_text()
    # print(len(normalized_data))

    tokenizer = BertTokenizer.from_pretrained(
        # 'bert-base-uncased', 'bert-base-multilingual-uncased'
        pretrained_model_name_or_path='bert-base-multilingual-cased',
        cache_dir='../',
        do_lower_case=True,
        max_len=1e12,
        do_basic_tokenize=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]",
                     "[MASK]", "[TLE]", "[DOC]", "[PAR]")
    )

    data_handler.retrieve_max_contexlen(tokenizer)

    exit(0)

    sentence = 'to whom did virgin mary allegedly appear in 1858 in lourdes france'

    #sentence = 'table tr th 2016 rank th th city th th state th th 2016 estimate th th census th th change th th colspan2 2016 land area th th colspan2 2016 population density th th location th tr tr td td td new york td td new york td td 8537673 td td 8175133 td td 7000443466791304800 ♠ 443 td td 3015 sq mi td td 7809 km td td 28317 sq mi td td 10933 km td td 40 ° 39 ′ 49 n 73 ° 56 ′ 19 w ﻿ ﻿ 406635 ° n 739387 ° w ﻿ 406635 739387 ﻿ 1 new york city td tr tr td td td los angeles td td california td td 3976322 td td 3792621 td td 7000484364243092050 ♠ 484 td td 4687 sq mi td td 12139 km td td 8484 sq mi td td 3276 km td td 34 ° 01 ′ 10 n 118 ° 24 ′ 39 w ﻿ ﻿ 340194 ° n 1184108 ° w ﻿ 340194 1184108 ﻿ 2 los angeles td tr tr td td td chicago td td illinois td td 2704958 td td 2695598 td td 6999347232784710470 ♠ 035 td td 2273 sq mi td td 5887 km td td 11900 sq mi td td 4600 km td td 41 ° 50 ′ 15 n 87 ° 40 ′ 54 w ﻿ ﻿ 418376 ° n 876818 ° w ﻿ 418376 876818 ﻿ 3 chicago td tr tr td td td houston td td texas td td 2303482 td td 2100263 td td 7000967588344888229 ♠ 968 td td 6375 sq mi td td 16511 km td td 3613 sq mi td td 1395 km td td 29 ° 47 ′ 12 n 95 ° 23 ′ 27 w ﻿ ﻿ 297866 ° n 953909 ° w ﻿ 297866 953909 ﻿ 4 houston td tr tr td 5 td td phoenix td td arizona td td 1615017 td td 1445632 td td 7001117170206525590 ♠ 1172 td td 5176 sq mi td td 13406 km td td 3120 sq mi td td 1200 km td td 33 ° 34 ′ 20 n 112 ° 05 ′ 24 w ﻿ ﻿ 335722 ° n 1120901 ° w ﻿ 335722 1120901 ﻿ 6 phoenix td tr tr td 6 td td philadelphia td td pennsylvania td td 1567872 td td 1526006 td td 7000274350166382040 ♠ 274 td td 1342 sq mi td td 3476 km td td 11683 sq mi td td 4511 km td td 40 ° 00 ′ 34 n 75 ° 08 ′ 00 w ﻿ ﻿ 400094 ° n 751333 ° w ﻿ 400094 751333 ﻿ 5 philadelphia td tr tr td 7 td td san antonio td td texas td td 1492510 td td 1327407 td td 7001124380088397910 ♠ 1244 td td 4610 sq mi td td 11940 km td td 3238 sq mi td td 1250 km td td 29 ° 28 ′ 21 n 98 ° 31 ′ 30 w ﻿ ﻿ 294724 ° n 985251 ° w ﻿ 294724 985251 ﻿ 7 san antonio td tr tr td 8 td td san diego td td california td td 1406630 td td 1307402 td td 7000758970844468650 ♠ 759 td td 3252 sq mi td td 8423 km td td 4325 sq mi td td 1670 km td td 32 ° 48 ′ 55 n 117 ° 08 ′ 06 w ﻿ ﻿ 328153 ° n 1171350 ° w ﻿ 328153 1171350 ﻿ 8 san diego td tr tr td 9 td td dallas td td texas td td 1317929 td td 1197816 td td 7001100276670206440 ♠ 1003 td td 3409 sq mi td td 8829 km td td 3866 sq mi td td 1493 km td td 32 ° 47 ′ 36 n 96 ° 45 ′ 59 w ﻿ ﻿ 327933 ° n 967665 ° w ﻿ 327933 967665 ﻿ 9 dallas td tr tr td 10 td td san jose td td california td td 1025350 td td 945942 td td 7000839459501745350 ♠ 839 td td 1775 sq mi td td 4597 km td td 5777'

    sentence = "p east west schism also called great schism and schism of 1054 was break of communion between what are now eastern orthodox and roman catholic churches which has lasted since 11th century it is not to be confused with western schism which lasted from 1378 to 1417 which is also sometimes called great schism p"

    tokens = tokenizer.tokenize(sentence)

    print('Sentence: \n', sentence)
    print('Tokens: ', tokens)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print('Tokens_ids: ', token_ids)

    reconstructed_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print('Reconstructed_tokens: ', reconstructed_tokens)

    reconstructed_sentence = DatasetHandler.reconstruct_text_from_tokens(
        reconstructed_tokens)
    print('Reconstructed_sentence: \n', reconstructed_sentence)

    print("Equality: ", reconstructed_sentence == sentence)

    print('vocab size: ', len(tokenizer.vocab))

    print('done!')
