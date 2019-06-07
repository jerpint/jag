import numpy as np
import json

try:
    from .transformer import TransformerEncoder
except ModuleNotFoundError:
    from transformer import TransformerEncoder


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


if __name__ == '__main__':

    model = load_openai_transformer(
        path='./cache/pre_trained/openai/model/',
        special_count=5,
        num_segments=2,
        use_pooler=True, use_masked_lm=False, use_next_sp=False,
        do_seq_class_task=False, do_mult_choice_task=False,
        do_tok_class_task=False, do_qa_task=False,
        seq_class_num_labels=2, task_num_choices=2, tok_class_num_labels=2,
        task_dropout=0.1
    )
    print("done!")
