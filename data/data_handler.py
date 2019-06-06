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
import six
import collections

from .bert_tokenization import BertTokenizer


def printable_text(text):
    """Returns text encoded in a way suitable for print."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.

    if six.PY3:
        unicode = str
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def remove_html_tags(text):
    return re.sub(r'<(.|\n)*?>', ' ', text)


def remove_punc(text):
    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    exclude = set(string.punctuation)
    exclude.add('’')
    exclude.add('“')
    exclude.add('”')
    exclude.add('‘')
    return ''.join(ch for ch in text if ch not in exclude)


def white_space_fix(text):
    return ' '.join(text.strip().split())


def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def lower(text):
    return text.lower()


def normalize(s: str) -> str:
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class MRQAExample(object):
    """A single training/test example.
       For examples without an answer,
       the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 question_tokens,
                 context_text,
                 context_tokens,
                 ds_id=None,
                 answers_text=None,
                 contextualized_answers=None,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.ds_id = ds_id
        self.question_text = question_text
        self.question_tokens = question_tokens
        self.context_text = context_text
        self.context_tokens = context_tokens
        self.answers_text = answers_text
        self.contextualized_answers = contextualized_answers
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (printable_text(self.qas_id))
        s += ", question_text: %s" % (
            printable_text(self.question_text))
        s += ", context_tokens: [%s]" % (" ".join(self.context_tokens))
        if self.start_position:
            s += ", start_position: {}".format(self.start_position)
        if self.end_position:
            s += ", end_position: {}".format(self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 ctx_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 ds_id=None,
                 qas_id=None,
                 start_position=None,
                 end_position=None,
                 token_classes=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.ds_id = ds_id
        self.qas_id = qas_id
        self.example_index = example_index
        self.ctx_span_index = ctx_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.token_classes = token_classes
        self.is_impossible = is_impossible


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)
        self.ds_id_map = dict()
        self.qua_id_map = dict()

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        def create_float_feature(values):
            feature = tf.train.Feature(
                float_list=tf.train.FloatList(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["example_index"] = create_int_feature([feature.example_index])
        features["ctx_span_index"] = create_int_feature(
            [feature.ctx_span_index]
        )
        if str(feature.ds_id) not in self.ds_id_map:
            self.ds_id_map[str(feature.ds_id)] = len(self.ds_id_map)
        features["ds_id"] = create_int_feature(
            [self.ds_id_map[str(feature.ds_id)]]
        )
        if str(feature.qas_id) not in self.qua_id_map:
            self.qua_id_map[str(feature.qas_id)] = len(self.qua_id_map)
        features["qas_id"] = create_int_feature(
            [self.qua_id_map[str(feature.qas_id)]]
        )
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature(
                feature.start_position
            )
            features["end_positions"] = create_int_feature(
                feature.end_position
            )
            features["token_classes"] = create_int_feature(
                feature.token_classes
            )
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def input_fn_builder(
        input_file, seq_length, max_answer_num, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to our model."""

    # def init_func(_):
    #     return 0.0
    # def reduce_func(state, value):
    #     return state + value['features']
    # def finalize_func(state):
    #     return state
    # reducer = tf.data.experimental.Reducer(
    #     init_func, reduce_func, finalize_func
    # )

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "example_index": tf.FixedLenFeature([], tf.int64),
        "ctx_span_index": tf.FixedLenFeature([], tf.int64),
        "ds_id": tf.FixedLenFeature([], tf.int64),
        "qas_id": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature(
            [max_answer_num], tf.int64
        )
        name_to_features["end_positions"] = tf.FixedLenFeature(
            [max_answer_num], tf.int64
        )
        name_to_features["token_classes"] = tf.FixedLenFeature(
            [seq_length], tf.int64
        )
        name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


class DatasetHandler():
    """ Class with different tools for the dataset.

    """

    def __init__(self, data_src: str = 'mrqa_urls.txt', cache_dir: str = '.'):
        self.dpath_dict = self.get_file(data_src, cache_dir)
        print(self.dpath_dict)

    def preprocess(self, s: str) -> str:
        """Preprocess the string s. These operations are from the evaluation script
        of the MRQA challenge."""

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

    def read_mrqa_examples_from_dataset(
            self, dataset_name: str, is_training=True) -> List[MRQAExample]:
        """ Read data examples from a given dataset.
        If the dataset does not exists in cache_dir, it is downloaded.
        Args:
            dataset_path (str): path of the dataset
        Returns:
            examples: A list of MRQAExample data
        """
        if dataset_name in self.dpath_dict:
            f = gzip.open(self.dpath_dict[dataset_name], 'rb')
        else:
            f = gzip.open(dataset_name, 'rb')

        examples = []
        dataset_id = None

        special_tokens = set(['[TLE]', '[DOC]', '[PAR]'])

        escape_examples = 0

        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in ex:
                dataset_id = ex['header']['dataset']
                if 'split' in ex['header']:
                    split = ex['header']['split']
                else:
                    split = ex['header']['mrqa_split']
                is_training = (split in ('dev', 'train')) and is_training
                dataset_id += '_' + split
                continue

            raw_context = ex['context']
            raw_context_tokens = ex['context_tokens']

            # preprocess context
            context_tokens = []
            context_token_offset = []
            char_to_word_offset = []
            sum_offset = 0
            orig_token_map = []
            for k in range(len(raw_context_tokens)):
                token, _ = raw_context_tokens[k]

                if token not in special_tokens:
                    token = remove_html_tags(token).strip()
                    token = remove_punc(token).strip()

                if len(token) == 0:
                    orig_token_map.append(None)
                    continue

                context_tokens.append(token)
                context_token_offset.append(sum_offset)
                sum_offset += len(token) + 1  # for spaces
                char_to_word_offset.extend(
                    [len(context_tokens) - 1] * (len(token) + 1)
                )
                orig_token_map.append(len(context_tokens) - 1)

            context = ' '.join(context_tokens)
            context_lower = context.lower()

            for qa in ex['qas']:
                qa_id = qa['qid']
                raw_question = qa['question']
                raw_question_tokens = qa['question_tokens']

                question_tokens = []
                question_token_offset = []
                question_sum_offset = 0
                for k in range(len(raw_question_tokens)):
                    token, _ = raw_question_tokens[k]

                    if token not in special_tokens:
                        token = remove_html_tags(token).strip()
                        token = remove_punc(token).strip()

                    if len(token) == 0:
                        continue

                    question_tokens.append(token)
                    question_token_offset.append(question_sum_offset)
                    question_sum_offset += len(token) + 1  # for spaces

                question = ' '.join(question_tokens)
                _ = len(question)

                start_position = None
                end_position = None
                answers = None

                answer_list = []
                detected_answers_list = None
                effective_answer_list = []

                if is_training:
                    answer_list = qa['answers']
                    detected_answers_list = qa['detected_answers']

                    answers = []
                    start_position = []
                    end_position = []

                    for das_idx, das in enumerate(detected_answers_list):
                        raw_answer_text = das['text']
                        ts_beg, ts_end = das['token_spans'][0]
                        raw_answer_tokens = [
                            raw_context_tokens[k]
                            for k in range(ts_beg, ts_end + 1)
                        ]

                        answer_tokens = []
                        answer_token_offset = []
                        answer_sum_offset = 0
                        for k in range(len(raw_answer_tokens)):
                            token, _ = raw_answer_tokens[k]
                            if token not in special_tokens:
                                token = remove_html_tags(token).strip()
                                token = remove_punc(token).strip()

                            if len(token) == 0:
                                continue

                            answer_tokens.append(token)
                            answer_token_offset.append(answer_sum_offset)
                            answer_sum_offset += len(token) + 1  # for spaces

                        answer = ' '.join(answer_tokens)

                        answer_lower = answer.lower()

                        answer_lower_n = normalize(answer_lower)
                        raw_answer_lower_n = normalize(raw_answer_text)
                        _ = len(raw_answer_lower_n)

                        # if answer_lower_n != raw_answer_lower_n:
                        #     print(
                        #         "Could not find answer: '{}' vs. '{}' ".format(
                        #             answer_lower_n, raw_answer_lower_n
                        #         )
                        #     )
                        #     escape_examples += 1
                        #     exit()
                        #     continue

                        # find all occurences in the context

                        ans_start_pos = []
                        ans_end_pos = []
                        for k in range(len(das['token_spans'])):
                            ts_beg, ts_end = das['token_spans'][k]
                            new_beg = orig_token_map[ts_beg]
                            new_end = orig_token_map[ts_end]

                            t_b = ts_beg
                            t_e = ts_end
                            if new_beg is None:

                                while new_beg is None and t_b < len(orig_token_map):
                                    new_beg = orig_token_map[t_b]
                                    t_b += 1

                            if new_end is None:
                                while new_end is None and t_e > 0:
                                    new_end = orig_token_map[t_e]
                                    t_e -= 1

                            if new_beg is None or new_end is None:
                                escape_examples += 1
                                continue

                            ans_start_pos.append(new_beg)
                            ans_end_pos.append(new_end)

                            test_answer = ' '.join(
                                context_tokens[new_beg:new_end + 1])
                            test_answer_lower = test_answer.lower()

                            answer_lower_n = normalize(answer_lower)
                            test_answer_lower_n = normalize(test_answer_lower)

                            if answer_lower_n != test_answer_lower_n:
                                print(
                                    "Could not find answer: '{}' vs. '{}' ".format(
                                        answer_lower_n, test_answer_lower_n
                                    )
                                )
                                # print(list(enumerate(raw_context_tokens)))
                                # print('')
                                # print('')
                                # print(list(enumerate(context_tokens)))
                                # print('')
                                # print('')
                                # print(list(enumerate(orig_token_map)))
                                # print('')
                                # print('')
                                # print((ts_beg, ts_end), (t_b, t_e),
                                #       (new_beg, new_end))
                                # print(qa_id)

                                escape_examples += 1
                                continue

                        if len(ans_start_pos) > 0 and len(ans_end_pos) > 0:
                            start_position.append(ans_start_pos)
                            end_position.append(ans_end_pos)
                            effective_answer_list.append(answer_list[das_idx])
                            answers.append(answer)

                        # index = 0
                        # all_indices = []
                        # answer_len = len(answer_lower)
                        # while index < len(context_lower):
                        #     index = context_lower.find(answer_lower, index)
                        #     if index == -1:
                        #         break
                        #     all_indices.append(index)
                        #     index += answer_len
                        # if len(all_indices) == 0:
                        #     print(
                        #       "Could not find answer in context: '{}' vs. '{}'".format(
                        #           answer_lower, context_lower
                        #       )
                        #     )
                        #     continue
                        # answers.append(answer)
                        # start_position.append(
                        #     [char_to_word_offset[k] for k in all_indices]
                        # )
                        # end_position.append(
                        #     [
                        #         char_to_word_offset[k + answer_len - 1]
                        #         for k in all_indices
                        #     ]
                        # )

                    if len(answers) == 0:
                        print(
                            "Could not find any answer in context: '{}' vs. '{}'".format(
                                answer_list, context_lower
                            )
                        )
                        continue

                example = MRQAExample(
                    ds_id=dataset_id,
                    qas_id=qa_id,
                    question_text=raw_question,
                    question_tokens=question_tokens,
                    context_text=raw_context,
                    context_tokens=context_tokens,
                    answers_text=effective_answer_list,  # answer_list,
                    contextualized_answers=answers,
                    start_position=start_position,
                    end_position=end_position,
                )
                examples.append(example)

        print('escape examples: ', escape_examples)
        return examples

    def read_mrqa_examples(self, is_training=True) -> List[MRQAExample]:
        """ Retrieve the examples from the provided datasets
        Args:
        Returns:
            examples: A list of MRQAExample data
        """
        examples = []
        for dataset_name in self.dpath_dict.keys():
            print('Processing  dataset: {}'.format(dataset_name))
            data = self.read_mrqa_examples_from_dataset(
                dataset_name, is_training
            )
            examples += data
            print('Example length so far: {:d}'.format(len(examples)))
        print('Processing completed')
        return examples

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring
        # documents, a single token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C.
        # We only want to consider the score with "maximum context",
        # which we define as the *minimum* of its left and right context
        # (the *sum* of left and right context will always be the same,
        # of course).
        #
        # In the example the maximum context for 'bought' would be
        # span C since it has 1 left context and 3 right context, while
        # span B has 4 left context and 0 right context.

        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + \
                0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _improve_answer_span(
            self, doc_tokens, input_start, input_end,
            tokenizer, orig_answer_text):
        """Returns tokenized answer spans that better match
           the annotated answer.
        """

        # The SQuAD annotations are character based. We first project
        # them to whitespace-tokenized words. But then after WordPiece
        # tokenization, we can often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).".
        # However after tokenization, our tokens will be "( 1895 - 1943 ) .".
        # So we can match the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in
        #            the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span
        # of the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation.
        # This is fairly rare in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def convert_examples_to_features(
            self, examples, tokenizer, max_seq_length,
            ctx_stride, max_query_length, max_answer_num, output_fn,
            keep_partial_answer_span=False,
            same_token_class_per_answer_token=True,
            unique_id_start=1000000000):
        """ Convert examples into a list of `InputFeatures`.

        Args:
            examples (List[MRQAExample]): List of examples to convert
            tokenizer: tokenizer to be used
            max_seq_length (int): maximum length of the context sequence
            ctx_stride (int): stride to be used when splitting
        the context sequence
            max_query_length (int): maximum length of the question sequence
            max_answer_num (int): maximum occurence number of answers
        for a query
            output_fn: callback function
            keep_partial_answer_span (bool): wheter to keep partial answer
        spans in the chunks or not
            unique_id_start: start of the feature ids being generated


        Returns:
            features (List[InputFeatures]): A list of input feature data

        """

        results = []
        unique_id = unique_id_start
        for example_idx, example in enumerate(examples):
            query_text = ' '.join(example.question_tokens)
            query_tokens = tokenizer.tokenize(query_text)

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            # decompose original tokens into subtokens using the tokenizer
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_context_tokens = []
            for (i, token) in enumerate(example.context_tokens):
                orig_to_tok_index.append(len(all_context_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_context_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None

            is_training = ((example.start_position is not None) and
                           (example.end_position is not None))
            is_impossible = is_training and (
                len(example.start_position) == 0 and len(
                    example.end_position) == 0
            )

            if is_training and is_impossible:
                tok_start_position = [[-1]]
                tok_end_position = [[-1]]

            if is_training and not is_impossible:
                tok_start_position = []
                tok_end_position = []
                for i in range(len(example.answers_text)):
                    ans_start_pos = []
                    ans_end_pos = []
                    for k_idx in range(len(example.start_position[i])):
                        start_pos = orig_to_tok_index[
                            example.start_position[i][k_idx]]
                        if (
                            example.end_position[i][
                                k_idx] < len(example.context_tokens) - 1
                        ):
                            end_pos = orig_to_tok_index[
                                example.end_position[i][k_idx] + 1] - 1
                        else:
                            end_pos = len(all_context_tokens) - 1

                        start_pos, end_pos = self._improve_answer_span(
                            all_context_tokens, start_pos, end_pos, tokenizer,
                            example.answers_text[i]
                        )
                        ans_start_pos.append(start_pos)
                        ans_end_pos.append(end_pos)

                    tok_start_position.append(ans_start_pos)
                    tok_end_position.append(ans_end_pos)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_context = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence
            # length. To deal with this we do a sliding window approach,
            # where we take chunks of the up to our max length with a stride
            # of `ctx_stride`.

            CtxSpan = collections.namedtuple("CtxSpan", ["start", "length"])
            ctx_spans = []
            start_offset = 0
            while start_offset < len(all_context_tokens):
                length = len(all_context_tokens) - start_offset
                if length > max_tokens_for_context:
                    length = max_tokens_for_context
                ctx_spans.append(CtxSpan(start=start_offset, length=length))
                if start_offset + length >= len(all_context_tokens):
                    break
                start_offset += min(length, ctx_stride)

            for (ctx_span_index, ctx_span) in enumerate(ctx_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                token_classes = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                token_classes.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                    token_classes.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)
                token_classes.append(0)

                for i in range(ctx_span.length):
                    split_token_index = ctx_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[
                        split_token_index
                    ]

                    is_max_context = self._check_is_max_context(
                        ctx_spans, ctx_span_index, split_token_index
                    )
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_context_tokens[split_token_index])
                    segment_ids.append(1)
                    token_classes.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)
                token_classes.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # The mask has 1 for real tokens and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    token_classes.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(token_classes) == max_seq_length

                start_position = None
                end_position = None
                out_of_span = False
                if is_training and not is_impossible:
                    # For training, if our document chunk does not contain
                    # an annotation we throw it out, since there is nothing
                    # to predict.
                    ctx_start = ctx_span.start
                    ctx_end = ctx_span.start + ctx_span.length - 1

                    valid_tok_start_pos = []
                    valid_tok_end_pos = []

                    for k in range(len(tok_start_position)):
                        for d in range(len(tok_start_position[k])):
                            if (tok_start_position[k][d] >= ctx_start and
                                    tok_end_position[k][d] <= ctx_end):
                                valid_tok_start_pos.append(
                                    tok_start_position[k][d]
                                )
                                valid_tok_end_pos.append(
                                    tok_end_position[k][d]
                                )
                            else:
                                if keep_partial_answer_span:
                                    if (tok_end_position[k][d] >= ctx_start
                                            and tok_end_position[k][d] <= ctx_end
                                            and tok_start_position[k][d] < ctx_start):
                                        valid_tok_start_pos.append(
                                            ctx_start
                                        )
                                        valid_tok_end_pos.append(
                                            tok_end_position[k][d]
                                        )
                                    elif (tok_start_position[k][d] >= ctx_start
                                            and tok_start_position[k][d] <= ctx_end
                                            and tok_end_position[k][d] > ctx_end):
                                        valid_tok_start_pos.append(
                                            tok_start_position[k][d]
                                        )
                                        valid_tok_end_pos.append(
                                            ctx_end
                                        )

                    if len(valid_tok_start_pos) == 0:
                        out_of_span = True
                    if out_of_span:
                        start_position = [0] * max_answer_num
                        end_position = [0] * max_answer_num
                    else:
                        ctx_offset = len(query_tokens) + 2
                        start_position = [
                            v_tok_s_pos - ctx_start + ctx_offset
                            for v_tok_s_pos in valid_tok_start_pos
                        ]
                        end_position = [
                            v_tok_e_pos - ctx_start + ctx_offset
                            for v_tok_e_pos in valid_tok_end_pos
                        ]
                        if len(start_position) > max_answer_num:
                            start_position = start_position[0:max_answer_num]
                            end_position = end_position[0:max_answer_num]
                        elif len(start_position) < max_answer_num:
                            tmp_len = len(start_position)
                            start_position += [0] * (max_answer_num - tmp_len)
                            end_position += [0] * (max_answer_num - tmp_len)
                        for k in range(len(start_position)):
                            tmp_s = start_position[k]
                            tmp_e = end_position[k]
                            if tmp_s == 0:
                                continue
                            for d in range(tmp_s, tmp_e + 1):
                                if d == tmp_s:
                                    token_classes[d] = 2
                                elif d == tmp_e:
                                    token_classes[d] = 2
                                else:
                                    if same_token_class_per_answer_token:
                                        token_classes[d] = 2
                                    else:
                                        token_classes[d] = 3

                if is_training and is_impossible:
                    start_position = [0] * max_answer_num
                    end_position = [0] * max_answer_num

                if example_idx < 20:
                    tf.logging.info("*** Example ***")
                    tf.logging.info("unique_id: %s" % (unique_id))
                    tf.logging.info("example_index: %s" % (example_idx))
                    tf.logging.info("ctx_span_index: %s" % (ctx_span_index))
                    tf.logging.info("tokens: %s" % " ".join(
                        [printable_text(x) for x in tokens]))
                    tf.logging.info("token_to_orig_map: %s" % " ".join(
                        ["%d:%d" % (x, y)
                         for (x, y) in six.iteritems(token_to_orig_map)
                         ]
                    ))
                    tf.logging.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y)
                        for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    tf.logging.info(
                        "input_ids: %s" % " ".join([str(x) for x in input_ids])
                    )
                    tf.logging.info(
                        "input_mask: %s" % " ".join(
                            [str(x) for x in input_mask]
                        )
                    )
                    tf.logging.info(
                        "segment_ids: %s" % " ".join(
                            [str(x) for x in segment_ids]
                        )
                    )
                    tf.logging.info(
                        "token_classes: %s" % " ".join(
                            [str(x) for x in token_classes]
                        )
                    )
                    if is_training and is_impossible:
                        tf.logging.info("impossible example")
                    if is_training and not is_impossible:
                        if out_of_span:
                            tf.logging.info("Out of Span")
                        else:
                            for k in range(len(start_position)):
                                answer_text = " ".join(
                                    tokens[start_position[k]:(end_position[k] + 1)])
                                tf.logging.info("start_position_%d: %d" %
                                                (k, start_position[k]))
                                tf.logging.info("end_position_%d: %d" %
                                                (k, end_position[k]))
                                tf.logging.info(
                                    "answer_%d: %s" %
                                    (k, printable_text(answer_text)))

                feature = InputFeatures(
                    unique_id=unique_id,
                    ds_id=example.ds_id,
                    qas_id=example.qas_id,
                    example_index=example_idx,
                    ctx_span_index=ctx_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_classes=token_classes,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible
                )

                # Run callback
                if output_fn is not None:
                    output_fn(feature)
                results.append(feature)

                unique_id += 1

        return results

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

    exit()

    data_handler = DatasetHandler(cache_dir=args.cache_dir)
    data_handler.read_mrqa_examples()

    exit()

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
