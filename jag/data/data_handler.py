import gzip
import json

from jag.data.mrqa_example import MRQAExample
from jag.utils.data_fetcher import get_file
from jag.utils.data_utils import remove_html_tags, remove_punc, normalize


class DatasetHandler():
    """ Class for reading MRQA dataset files.
        Args:
            data_src (str): path to a file containing the list of
                MRQA dataset files
            cache_dir (str): the directory in which the dataset files
                are or will be stored
    """

    def __init__(self, data_src='mrqa_urls.txt', cache_dir='.'):

        self.dpath_dict = get_file(data_src, cache_dir)
        print(self.dpath_dict)

    def read_mrqa_examples(
            self, is_training=True,
            allow_questions_with_no_answer=False):
        """ Retrieve the examples from the provided datasets
        Args:
            is_training (bool): a flag indicating whether or not the
                files are read for training purposes. Default: True.
            allow_questions_with_no_answer (bool): a flag indicating whether
                or not to skip examples with no answers. Default: False.
        Returns:
            examples: A list of MRQAExample data
        """
        examples = []
        for dataset_name in self.dpath_dict.keys():
            print('Processing  dataset: {}'.format(dataset_name))
            data = self.read_mrqa_examples_from_dataset(
                dataset_name, is_training, allow_questions_with_no_answer
            )
            examples += data
            print('Example length so far: {:d}'.format(len(examples)))
        print('Processing completed')
        return examples

    def read_mrqa_examples_from_dataset(
            self, dataset_name,
            is_training=True,
            allow_questions_with_no_answer=False):
        """ Read data examples from a given dataset.
        Args:
            dataset_path (str): path of the dataset
            is_training (bool): a flag indicating whether or not the
                files are read for training purposes. Default: True.
            allow_questions_with_no_answer (bool): a flag indicating whether
                or not to skip examples with no answers. Default: False.
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

                                escape_examples += 1
                                continue

                        if len(ans_start_pos) > 0 and len(ans_end_pos) > 0:
                            start_position.append(ans_start_pos)
                            end_position.append(ans_end_pos)
                            effective_answer_list.append(answer_list[das_idx])
                            answers.append(answer)

                    if len(answers) == 0:
                        print(
                            "Could not find any answer in context: '{}' vs. '{}'".format(
                                answer_list, context_lower
                            )
                        )
                        if not allow_questions_with_no_answer:
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
