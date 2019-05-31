import tensorflow as tf
import numpy as np
import json

from transformer import TransformerEncoder
from bert_modeling import BertConfig


def load_openai_transformer(
        path, use_attn_mask=True, max_len=512,
        use_one_embedding_dropout=False, **kwargs):

    with open(path + 'params_shapes.json') as f:
        shapes = json.load(f)

    SPECIAL_COUNT = 5
    NUM_SEGMENTS = 2

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
    init_params[1] = np.concatenate(
        (
            init_params[1],
            np.random.randn(SPECIAL_COUNT, 768).astype(np.float32) * 0.02
        ),
        axis=0
    )
    init_params = [
        np.zeros((NUM_SEGMENTS, 768)).astype(np.float32)
    ] + init_params  # segment embedding

    model = TransformerEncoder(
        vocab_size=40478 + SPECIAL_COUNT, n_layers=12, d_model=768,
        d_inner=768 * 4, n_head=12, d_k=768 // 12, d_v=768 // 12, d_out=768,
        max_len=min(512, max_len), num_segments=NUM_SEGMENTS,
        embedding_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
        embedding_layer_norm=False, layer_norm_epsilon=1e-5, neg_inf=-1e9,
        trainable_pos_embedding=True,
        use_one_embedding_dropout=use_one_embedding_dropout,
        use_attn_mask=use_attn_mask, use_pad_mask=False, use_gelu=True,
        accurate_gelu=False, **kwargs
    )
    maxi_len = min(512, max_len)
    input_shape = [
        (None, maxi_len),
        (None, maxi_len),
        (None, maxi_len)
    ]
    if use_attn_mask:
        input_shape.append(
            (None, 1, maxi_len, maxi_len)
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
        base_location, use_attn_mask=True, max_len=512,
        verbose=False, use_pooler=False, use_masked_lm=False,
        use_next_sp=False, **kwargs):

    if not use_pooler:
        use_next_sp = False

    BERT_SPECIAL_COUNT = 4
    BERT_UNUSED_COUNT = 99

    PAD_OFFSET = 0
    MSK_OFFSET = 1
    BOS_OFFSET = 2
    DEL_OFFSET = 3  # delimiter
    EOS_OFFSET = 4

    SPECIAL_COUNT = 5
    NUM_SEGMENTS = 2

    bert_config = BertConfig.from_json_file(
        base_location + 'bert_config.json'
    )
    init_checkpoint = base_location + 'bert_model.ckpt'
    var_names = tf.train.list_variables(init_checkpoint)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    vocab_size = bert_config.vocab_size - BERT_SPECIAL_COUNT - BERT_UNUSED_COUNT

    model = TransformerEncoder(
        vocab_size=vocab_size + SPECIAL_COUNT,
        n_layers=bert_config.num_hidden_layers,
        d_model=bert_config.hidden_size,
        d_inner=bert_config.intermediate_size,
        n_head=bert_config.num_attention_heads,
        d_k=bert_config.hidden_size // bert_config.num_attention_heads,
        d_v=bert_config.hidden_size // bert_config.num_attention_heads,
        d_out=bert_config.hidden_size,
        max_len=min(512, max_len),  # max_len,
        num_segments=NUM_SEGMENTS,
        embedding_dropout=0.1,
        attention_dropout=bert_config.attention_probs_dropout_prob,
        residual_dropout=bert_config.hidden_dropout_prob,
        embedding_layer_norm=True, layer_norm_epsilon=1e-12, neg_inf=-1e4,
        trainable_pos_embedding=True,
        use_one_embedding_dropout=True,
        use_attn_mask=use_attn_mask, use_pad_mask=False, use_gelu=True,
        accurate_gelu=True,  use_pooler=use_pooler, use_masked_lm=use_masked_lm,
        use_next_sp=use_next_sp,
        **kwargs
    )

    maxi_len = min(512, max_len)
    input_shape = [
        (None, maxi_len),
        (None, maxi_len),
        (None, maxi_len)
    ]
    if use_attn_mask:
        input_shape.append(
            (None, 1, maxi_len, maxi_len)
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
        first_vars_size = 5
        if parts[1] == 'embeddings':
            n = parts[-1]
            if n == 'token_type_embeddings':
                w_id = 0
            elif n == 'position_embeddings':
                w_id = 1
            elif n == 'word_embeddings':
                w_id = 2
            elif n == 'gamma':
                w_id = 3
            elif n == 'beta':
                w_id = 4
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

            if w_id == 1:  # pos embedding
                weights[w_id][:max_len, :] = check_point.get_tensor(
                    var_name
                )[:max_len, :] if not unsqueeze else check_point.get_tensor(
                    var_name
                )[None, :max_len, :]

            elif w_id == 2:  # word embedding
                # ours: unk, [vocab], pad, msk(mask), bos(cls),
                #       del(use sep again), eos(sep)
                # theirs: pad, 99 unused, unk, cls, sep, mask, [vocab]

                saved = check_point.get_tensor(
                    var_name)  # vocab_size, emb_size
                # weights[our_position] = saved[their_position]
                weights[w_id][0] = saved[1 + BERT_UNUSED_COUNT]  # unk
                weights[w_id][1:vocab_size] = saved[-vocab_size + 1:]
                weights[w_id][vocab_size + PAD_OFFSET] = saved[0]
                weights[w_id][vocab_size +
                              MSK_OFFSET] = saved[4 + BERT_UNUSED_COUNT]
                weights[w_id][vocab_size +
                              BOS_OFFSET] = saved[2 + BERT_UNUSED_COUNT]
                weights[w_id][vocab_size +
                              DEL_OFFSET] = saved[3 + BERT_UNUSED_COUNT]
                weights[w_id][vocab_size +
                              EOS_OFFSET] = saved[3 + BERT_UNUSED_COUNT]

            elif lm_flag:
                # ours: unk, [vocab], pad, msk(mask), bos(cls),
                #       del(use sep again), eos(sep)
                # theirs: pad, 99 unused, unk, cls, sep, mask, [vocab]

                saved = check_point.get_tensor(var_name)

                weights[w_id][0] = saved[1 + BERT_UNUSED_COUNT]  # unk
                weights[w_id][1:vocab_size] = saved[-vocab_size + 1:]
                weights[w_id][vocab_size + PAD_OFFSET] = saved[0]
                weights[w_id][vocab_size +
                              MSK_OFFSET] = saved[4 + BERT_UNUSED_COUNT]
                weights[w_id][vocab_size +
                              BOS_OFFSET] = saved[2 + BERT_UNUSED_COUNT]
                weights[w_id][vocab_size +
                              DEL_OFFSET] = saved[3 + BERT_UNUSED_COUNT]
                weights[w_id][vocab_size +
                              EOS_OFFSET] = saved[3 + BERT_UNUSED_COUNT]

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
    #     path='./pre_trained/openai/model/',
    #     use_pooler=True, use_masked_lm=False, use_next_sp=False,
    #     do_seq_class_task=False, do_mult_choice_task=False,
    #     do_tok_class_task=False, do_qa_task=False,
    #     seq_class_num_labels=2, task_num_choices=2, tok_class_num_labels=2,
    #     task_dropout=0.1
    # )

    model = load_google_bert(
        base_location='./pre_trained/google_bert/multi_cased_L-12_H-768_A-12/',
        verbose=True,
        use_pooler=True, use_masked_lm=False, use_next_sp=False,
        do_seq_class_task=False, do_mult_choice_task=False,
        do_tok_class_task=True, do_qa_task=False,
        seq_class_num_labels=2, task_num_choices=2, tok_class_num_labels=2,
        task_dropout=0.1
    )

    print('done!')
