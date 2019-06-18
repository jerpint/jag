import tensorflow as tf
import six
import collections
from jag.utils.data_utils import printable_text


class InputFeatures(object):
    """ A single set of features of data.

        Args:
            unique_id (int): the id of the input feature
            example_index (int): the index of the MRQAExample data
                in the loaded dataset.
            ctx_span_index (int): the index of the span within the MRQAExample
                from which the current feature is derived.
            tokens (List[str]): the tokens associated to this input feature
            token_to_orig_map (Dict[int, int]): a map of the current
                (sub-)tokens to the index of the original word from
                the textual context.
            token_is_max_context (Dict[int, bool]): a map of the current
                (sub-)tokens to boolean value indicating whether or not
                the current version of the token is the one having the maximum
                context. Due to the sliding window approach.
            input_ids (List[int]): The ids associated to the `tokens` data.
            input_mask (List[int]): The mask associated to the `tokens` data (Padding).
            segment_ids (List[int]): The segment ids associated to the `tokens` data.
            ds_id (str): id of the dataset from which the question is retrieved.
                Default: None.
            qas_id (str): the id of the question. Default: None
            start_position (List[int]): List of indexes corresponding to the start
                positions of a possible answer in `tokens`. Default: None
            end_position (List[int]): List of indexes corresponding to the end positions
                (aligned with the `start_position` elements) of a possible answer
                in `tokens`. Default: None
            token_classes (List[int]): The class associated to each token in `tokens`.
                Default: None
            is_impossible (bool): if True, the features represents an example
                with no answers. Default: None
    """

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


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.

        Args:
            doc_spans (List): List of all sliding windows (spans) of a context data
            cur_span_index (int): index of the sliding windows being considered.
            position (int): position of the token under evaluation within the
                considered sliding windows.
    """

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
        doc_tokens, input_start, input_end,
        tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match
       the annotated answer.

        Args:
            doc_tokens (List(str)]): List of the tokens of a context data
            input_start (int): the start index of an answer in `doc_tokens`.
            input_end (int): the end index of an answer in `doc_tokens`.
            tokenizer: a tokenizer to use
            orig_answer_text (str): the raw answer text
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
        examples, tokenizer, max_seq_length,
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
        output_fn: callback function with signature fn(InputFeature)
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

                    start_pos, end_pos = _improve_answer_span(
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

                is_max_context = _check_is_max_context(
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
