import numpy as np
import json
import tensorflow as tf

try:
    from .transformer import TransformerEncoder
    from .bert_config import BertConfig
except ModuleNotFoundError:
    from transformer import TransformerEncoder
    from bert_config import BertConfig


def load_openai_transformer(
        path,
        special_count=5,
        num_segments=2,
        use_attn_mask=True, max_len=512,
        use_one_embedding_dropout=False,
        is_training=True, **kwargs):
    r"""Load the pretrained weights of the OpenAI model.

    Inputs:
        ``path`` (str): the path containing the pretrained model
        ``special_count`` (int): the number of special tokens of
    your models.
        E.g., PAD, MSK, BOS, DEL, EOS. Default: 5
        ``num_segments`` (int): number of segments. if set to zero,
    then the segment
    embeddings  won't be performed. Default: 2.
        ``use_attn_mask`` (bool): whether or not the layer expects to use
    attention mask in the
    computation. Default: ``True``.
        ``max_len`` (int): maximum length of the input sequence. Default: 512.
        ``use_one_embedding_dropout``(bool): if ``True``, the different
    embeddings will be
    summed up before applying dropout, otherwise dropout will be applied
    to each embedding type independently before summing them.
    Default: ``False``.
        ``is_training`` (bool): whether or not the model is instantiated for
    training purposes

    Outputs:
        ``model``: the ``TransformerEncoder`` model instantiated with the
    pretrained weights

    """

    with open(path + 'params_shapes.json') as f:
        shapes = json.load(f)

    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [
        np.load(path + 'params_{}.npy'.format(n)) for n in range(10)
    ]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [
        param.reshape(shape)
        for param, shape in zip(init_params, shapes)
    ]
    init_params[0] = init_params[0][:min(512, max_len)]

    # add special token embedding to token embedding
    # the special tokens are added at the end of the [vocab]
    if special_count > 0:
        init_params[1] = np.concatenate(
            (
                init_params[1],
                np.random.randn(special_count, 768).astype(np.float32) * 0.02
            ),
            axis=0
        )
    if num_segments > 0:
        # adding parameters for segment embeddings if needed
        init_params = [
            np.zeros((num_segments, 768)).astype(np.float32)
        ] + init_params  # segment embedding

    kwargs['vocab_size'] = 40478 + special_count
    kwargs['n_layers'] = 12
    kwargs['d_model'] = 768
    kwargs['d_inner'] = 768 * 4
    kwargs['n_head'] = 12
    kwargs['d_k'] = 768 // 12
    kwargs['d_v'] = 768 // 12
    kwargs['d_out'] = 768
    kwargs['num_segments'] = num_segments
    kwargs['max_len'] = min(512, max_len)
    kwargs['embedding_layer_norm'] = False
    kwargs['trainable_pos_embedding'] = True

    if 'neg_inf' not in kwargs:
        kwargs['neg_inf'] = -1e9
    if 'layer_norm_epsilon' not in kwargs:
        kwargs['layer_norm_epsilon'] = 1e-5
    if 'embedding_dropout' not in kwargs:
        kwargs['embedding_dropout'] = 0.1
    if 'attention_dropout' not in kwargs:
        kwargs['attention_dropout'] = 0.1
    if 'residual_dropout' not in kwargs:
        kwargs['residual_dropout'] = 0.1
    if 'task_dropout' not in kwargs:
        kwargs['task_dropout'] = 0.1
    if 'use_gelu' not in kwargs:
        kwargs['use_gelu'] = True
    if 'accurate_gelu' not in kwargs:
        kwargs['accurate_gelu'] = False
    if 'use_pad_mask' not in kwargs:
        kwargs['use_pad_mask'] = False

    if not is_training:
        kwargs['embedding_dropout'] = 0.0
        kwargs['attention_dropout'] = 0.0
        kwargs['residual_dropout'] = 0.0
        kwargs['task_dropout'] = 0.0

    model = TransformerEncoder(
        use_one_embedding_dropout=use_one_embedding_dropout,
        use_attn_mask=use_attn_mask, **kwargs
    )
    maxi_len = min(512, max_len)
    input_shape = [(None, maxi_len)]
    if num_segments > 0:
        input_shape.append((None, maxi_len))
    input_shape.append((None, maxi_len))

    if use_attn_mask:
        input_shape.append(
            (None, 1, maxi_len, maxi_len)
        )
    if 'use_pad_mask' in kwargs and kwargs['use_pad_mask']:
        input_shape.append(
            (None, maxi_len, 1)
        )

    model.build(input_shape)

    n_params = len(init_params)

    weights = [None if i < n_params else w
               for i, w in enumerate(model.get_weights())]

    weights[:n_params] = init_params[:]

    # model.set_weights(init_params)
    model.set_weights(weights)

    return model


def load_google_bert(
        path,
        keep_all_bert_tokens=True,
        special_tokens=None,
        add_special_token_to_begin=True,
        num_segments=2,
        use_attn_mask=True, max_len=512,
        verbose=False, use_pooler=False, use_masked_lm=False,
        use_next_sp=False, is_training=True, **kwargs):
    r"""Load the pretrained weights of the Google BERT model. Nothe that
    their vocab is as: # ``pad, 99 unused, unk, cls, sep, mask, [vocab]``
    in case you may want to specify your own special token mapping

    Inputs:
        ``path`` (str): the path containing the pretrained model
        ``keep_all_bert_tokens``: whether or not to keep the original
    vocab embeddings of BERT with all its unused/special tokens
        ``special_tokens`` (List[(Token, Index_in_Bert_Vocab)]):
    the special tokens of mapping for your problem. only take into account
    if `keep_all_bert_tokens` is False.
    E.g., ('PAD', 0), ('MSK', 103), ('BOS', 101), ('DEL', 102), ('EOS', 102)
    Default: None
        ``add_special_token_to_begin``: if True, add the special token at
    the begin so that vocab as [special_tokens, [vocab]] otherwise we have
    [[vocab], special_tokens]. only take into account if
    `keep_all_bert_tokens` is False.
        ``num_segments`` (int): number of segments. if set to zero,
    then the segment
    embeddings  won't be performed. Default: 2.
        ``use_attn_mask`` (bool): whether or not the layer expects to use
    attention mask in the
    computation. Default: ``True``.
        ``max_len`` (int): maximum length of the input sequence. Default: 512.
        ``use_pooler`` (bool): whether or not to compute the pooled
    representation of the input sequnces. Default: ``False``.
        ``use_masked_lm`` (bool): whether or not to compute the masked
    language modeling outputs. Default: ``False``.
        ``use_next_sp`` (bool): whether or not to compute the outputs
    of the next sentence prediction task. Default: ``False``.
        ``is_training`` (bool): whether or not the model is instantiated for
    training purposes

    Outputs:
        ``model``: the ``TransformerEncoder`` model instantiated with the
    pretrained weights

    """

    if not use_pooler:
        use_next_sp = False

    BERT_SPECIAL_COUNT = 4
    BERT_UNUSED_COUNT = 99

    if special_tokens is None:
        special_tokens = []
    special_count = len(special_tokens)

    bert_config = BertConfig.from_json_file(
        path + 'bert_config.json'
    )
    init_checkpoint = path + 'bert_model.ckpt'
    var_names = tf.train.list_variables(init_checkpoint)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    if keep_all_bert_tokens:
        vocab_size = bert_config.vocab_size - special_count
    else:
        vocab_size = bert_config.vocab_size - BERT_SPECIAL_COUNT - BERT_UNUSED_COUNT

    if 'neg_inf' not in kwargs:
        kwargs['neg_inf'] = float(-1e4)
    if 'use_one_embedding_dropout' not in kwargs:
        kwargs['use_one_embedding_dropout'] = True
    if 'layer_norm_epsilon' not in kwargs:
        kwargs['layer_norm_epsilon'] = 1e-12
    if 'embedding_dropout' not in kwargs:
        kwargs['embedding_dropout'] = 0.1
    if 'attention_dropout' not in kwargs:
        kwargs['attention_dropout'] = bert_config.attention_probs_dropout_prob
    if 'residual_dropout' not in kwargs:
        kwargs['residual_dropout'] = bert_config.hidden_dropout_prob
    if 'task_dropout' not in kwargs:
        kwargs['task_dropout'] = 0.1
    if 'use_gelu' not in kwargs:
        kwargs['use_gelu'] = True
    if 'accurate_gelu' not in kwargs:
        kwargs['accurate_gelu'] = True
    if 'use_pad_mask' not in kwargs:
        kwargs['use_pad_mask'] = False
    kwargs['vocab_size'] = vocab_size + special_count
    kwargs['n_layers'] = bert_config.num_hidden_layers
    kwargs['d_model'] = bert_config.hidden_size
    kwargs['d_inner'] = bert_config.intermediate_size
    kwargs['n_head'] = bert_config.num_attention_heads
    kwargs['d_k'] = bert_config.hidden_size // bert_config.num_attention_heads
    kwargs['d_v'] = bert_config.hidden_size // bert_config.num_attention_heads
    kwargs['d_out'] = bert_config.hidden_size
    kwargs['num_segments'] = num_segments
    kwargs['max_len'] = min(512, max_len)
    kwargs['embedding_layer_norm'] = True
    kwargs['trainable_pos_embedding'] = True

    if not is_training:
        kwargs['embedding_dropout'] = 0.0
        kwargs['attention_dropout'] = 0.0
        kwargs['residual_dropout'] = 0.0
        kwargs['task_dropout'] = 0.0

    model = TransformerEncoder(
        use_attn_mask=use_attn_mask,
        use_pooler=use_pooler, use_masked_lm=use_masked_lm,
        use_next_sp=use_next_sp,
        **kwargs
    )

    maxi_len = min(512, max_len)
    input_shape = [(None, maxi_len)]
    if num_segments > 0:
        input_shape.append((None, maxi_len))
    input_shape.append((None, maxi_len))
    if use_attn_mask:
        input_shape.append(
            (None, 1, maxi_len, maxi_len)
        )
    if 'use_pad_mask' in kwargs and kwargs['use_pad_mask']:
        input_shape.append(
            (None, maxi_len, 1)
        )

    model.build(input_shape)

    # weights = [np.zeros(w.shape) for w in model.weights]
    weights = [w for i, w in enumerate(model.get_weights())]
    if verbose:
        print('weight num: ', len(weights))

    for var_name, _ in var_names:
        w_id = None
        qkv = None
        unsqueeze = False
        transpose = False
        lm_flag = False
        parts = var_name.split('/')
        beg_off = 0 if num_segments > 0 else -1  # no segments
        first_vars_size = 5 + beg_off
        if parts[1] == 'embeddings':
            n = parts[-1]
            if n == 'token_type_embeddings':
                if num_segments <= 0:
                    continue
                w_id = 0 + beg_off
            elif n == 'position_embeddings':
                w_id = 1 + beg_off
            elif n == 'word_embeddings':
                w_id = 2 + beg_off
            elif n == 'gamma':
                w_id = 3 + beg_off
            elif n == 'beta':
                w_id = 4 + beg_off
            else:
                raise ValueError()
        elif parts[2].startswith('layer_'):
            layer_number = int(parts[2][len('layer_'):])
            if parts[3] == 'attention':
                if parts[-1] == 'beta':
                    w_id = first_vars_size + layer_number * 12 + 5
                elif parts[-1] == 'gamma':
                    w_id = first_vars_size + layer_number * 12 + 4
                elif parts[-2] == 'dense':
                    if parts[-1] == 'bias':
                        w_id = first_vars_size + layer_number * 12 + 3
                    elif parts[-1] == 'kernel':
                        w_id = first_vars_size + layer_number * 12 + 2
                        unsqueeze = True
                    else:
                        raise ValueError()
                elif(
                    (parts[-2] == 'key') or (parts[-2] == 'query') or
                    (parts[-2] == 'value')
                ):
                    tmp = (0 if parts[-1] == 'kernel' else 1)
                    w_id = first_vars_size + layer_number * 12 + tmp
                    unsqueeze = parts[-1] == 'kernel'
                    qkv = parts[-2][0]
                else:
                    raise ValueError()
            elif parts[3] == 'intermediate':
                if parts[-1] == 'bias':
                    w_id = first_vars_size + layer_number * 12 + 7
                elif parts[-1] == 'kernel':
                    w_id = first_vars_size + layer_number * 12 + 6
                    unsqueeze = True
                else:
                    raise ValueError()
            elif parts[3] == 'output':
                if parts[-1] == 'beta':
                    w_id = first_vars_size + layer_number * 12 + 11
                elif parts[-1] == 'gamma':
                    w_id = first_vars_size + layer_number * 12 + 10
                elif parts[-1] == 'bias':
                    w_id = first_vars_size + layer_number * 12 + 9
                elif parts[-1] == 'kernel':
                    w_id = first_vars_size + layer_number * 12 + 8
                    unsqueeze = True
                else:
                    raise ValueError()
        elif parts[1] == 'pooler':
            if use_pooler:
                layer_number = bert_config.num_hidden_layers
                if parts[-1] == 'bias':
                    w_id = first_vars_size + layer_number * 12 + 1
                elif parts[-1] == 'kernel':
                    w_id = first_vars_size + layer_number * 12
                    unsqueeze = True
                else:
                    raise ValueError()
        elif parts[1] == 'predictions':
            layer_number = bert_config.num_hidden_layers
            base_offset = first_vars_size + layer_number * 12 + (
                2 if use_pooler else 0
            )
            if use_masked_lm:
                if parts[-1] == 'output_bias':
                    w_id = base_offset
                    lm_flag = True
                elif parts[-1] == 'gamma':
                    w_id = base_offset + 1
                elif parts[-1] == 'beta':
                    w_id = base_offset + 2
                elif parts[-1] == 'bias':
                    w_id = base_offset + 4
                elif parts[-1] == 'kernel':
                    w_id = base_offset + 3
                    unsqueeze = True
                else:
                    raise ValueError()
        elif parts[1] == 'seq_relationship':
            layer_number = bert_config.num_hidden_layers
            base_offset = first_vars_size + layer_number * 12 + (
                2 if use_pooler else 0
            )
            if use_masked_lm:
                base_offset += 6
            if use_next_sp:
                if parts[-1] == 'output_bias':
                    w_id = base_offset + 1
                elif parts[-1] == 'output_weights':
                    w_id = base_offset
                    unsqueeze = False
                    transpose = True
                else:
                    raise ValueError()

        if w_id is not None and qkv is None:
            if verbose:
                print('w_id: ', w_id)
                print(var_name, ' -> ', model.weights[w_id].name)

            tr_id = w_id - beg_off

            if tr_id == 0:  # segment embedding
                if num_segments > 0:
                    num_seg = min(num_segments, 2)
                    weights[w_id][:num_seg, :] = check_point.get_tensor(
                        var_name
                    )[:num_seg, :] if not unsqueeze else check_point.get_tensor(
                        var_name
                    )[None, :num_seg, :]

            elif tr_id == 1:  # pos embedding
                weights[w_id][:max_len, :] = check_point.get_tensor(
                    var_name
                )[:max_len, :] if not unsqueeze else check_point.get_tensor(
                    var_name
                )[None, :max_len, :]

            elif tr_id == 2:  # word embedding
                # ours: unk, [vocab], pad, msk(mask), bos(cls),
                #       del(use sep again), eos(sep)
                # theirs: pad, 99 unused, unk, cls, sep, mask, [vocab]

                # vocab_size, emb_size
                saved = check_point.get_tensor(var_name)

                if keep_all_bert_tokens:
                    weights[w_id][:] = saved
                else:
                    weights_vocab = saved[-vocab_size:]
                    if special_count > 0:
                        for i in range(len(special_tokens)):
                            idx = i
                            if not add_special_token_to_begin:
                                idx += vocab_size
                            assert special_tokens[i][
                                1] <= 103 and special_tokens[i][1] >= 0
                            weights[w_id][idx] = saved[special_tokens[i][1]]
                    if not add_special_token_to_begin:
                        idx = 0
                    else:
                        idx = special_count
                    weights[w_id][idx:vocab_size + idx] = weights_vocab

            elif lm_flag:
                # ours: unk, [vocab], pad, msk(mask), bos(cls),
                #       del(use sep again), eos(sep)
                # theirs: pad, 99 unused, unk, cls, sep, mask, [vocab]

                saved = check_point.get_tensor(var_name)

                if keep_all_bert_tokens:
                    weights[w_id][:] = saved
                else:
                    weights_vocab = saved[-vocab_size:]
                    if special_count > 0:
                        for i in range(len(special_tokens)):
                            idx = i
                            if not add_special_token_to_begin:
                                idx += vocab_size
                            assert special_tokens[i][
                                1] <= 103 and special_tokens[i][1] >= 0
                            weights[w_id][idx] = saved[
                                special_tokens[i][1]]
                    if not add_special_token_to_begin:
                        idx = 0
                    else:
                        idx = special_count
                    weights[w_id][idx:vocab_size + idx] = weights_vocab

            else:
                if not transpose:
                    weights[w_id][:] = check_point.get_tensor(
                        var_name
                    ) if not unsqueeze else check_point.get_tensor(
                        var_name
                    )[None, ...]
                else:
                    w_temp = check_point.get_tensor(
                        var_name
                    ) if not unsqueeze else check_point.get_tensor(
                        var_name
                    )[None, ...]
                    weights[w_id][:] = np.transpose(w_temp)

        elif w_id is not None:
            if verbose:
                print('w_id: ', w_id)
                print(var_name, ' -> ', model.weights[w_id].name, '::', qkv)

            p = {'q': 0, 'k': 1, 'v': 2}[qkv]
            if weights[w_id].ndim == 3:
                dim_size = weights[w_id].shape[1]
                weights[w_id][
                    0, :, p * dim_size:(p + 1) * dim_size
                ] = check_point.get_tensor(
                    var_name
                ) if not unsqueeze else check_point.get_tensor(
                    var_name
                )[None, ...]
            else:
                dim_size = weights[w_id].shape[0] // 3
                weights[w_id][
                    p * dim_size:(p + 1) * dim_size
                ] = check_point.get_tensor(var_name)
        else:
            if verbose:
                # TODO cls/predictions, cls/seq_relationship
                print('not mapped: ', var_name)
    model.set_weights(weights)
    return model


if __name__ == '__main__':

    # model = load_openai_transformer(
    #     path='./cache/pre_trained/openai/model/',
    #     special_count=5,
    #     num_segments=2,
    #     use_pooler=True, use_masked_lm=False, use_next_sp=False,
    #     do_seq_class_task=False, do_mult_choice_task=False,
    #     do_tok_class_task=False, do_qa_task=False,
    #     seq_class_num_labels=2, task_num_choices=2, tok_class_num_labels=2,
    #     task_dropout=0.1
    # )

    model = load_google_bert(
        path='./cache/pre_trained/google_bert/multi_cased_L-12_H-768_A-12/',
        verbose=True,
        keep_all_bert_tokens=False,
        special_tokens=[('PAD', 0), ('MSK', 103), ('BOS', 101),
                        ('DEL', 102), ('EOS', 102)],
        add_special_token_to_begin=False,
        num_segments=2,
        use_pooler=True, use_masked_lm=False, use_next_sp=False,
        do_seq_class_task=False, do_mult_choice_task=False,
        do_tok_class_task=True, do_qa_task=False,
        seq_class_num_labels=2, task_num_choices=2, tok_class_num_labels=2,
        task_dropout=0.1
    )
    print("done!")
