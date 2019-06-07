def test_transformer_shape():
    '''Basic test to make sure tensorflow is properly installed'''

    import tensorflow as tf
    import numpy as np
    from models.transformer import TransformerEncoder

    params = {
        'n_layers': 3,
        'd_inner': 50,
        'n_head': 5,
        'd_k': 15,
        'd_v': 15,
        'embedding_dropout': 0.1,
        'attention_dropout': 0.1,
        'residual_dropout': 0.1,
        'embedding_layer_norm': False,
        'layer_norm_epsilon': 1e-5,
        'neg_inf': float(-1e9),
        'trainable_pos_embedding': True,
        'use_one_embedding_dropout': True,
        'use_gelu': False,
        'accurate_gelu': False,
        'task_dropout': 0.1,
    }

    vocab_size = np.random.randint(20, 50)
    max_len = np.random.randint(10, 25)
    num_segments = 2
    d_model = np.random.randint(5, 20)
    d_out = d_model

    use_attn_mask = True if np.random.randint(0, 2) == 1 else False
    use_pad_mask = True if np.random.randint(0, 2) == 1 else False

    use_pooler = True if np.random.randint(0, 2) == 1 else False
    use_masked_lm = True if np.random.randint(0, 2) == 1 else False
    use_next_sp = True if np.random.randint(0, 2) == 1 else False

    do_seq_class_task = True if np.random.randint(0, 2) == 1 else False
    do_tok_class_task = True if np.random.randint(0, 2) == 1 else False
    do_qa_task = True if np.random.randint(0, 2) == 1 else False
    do_mult_choice_task = True if np.random.randint(0, 2) == 1 else False

    if do_seq_class_task or do_mult_choice_task:
        use_pooler = True

    # batch_size must be a multiple of this params
    task_num_choices = np.random.randint(2, 4)
    seq_class_num_labels = np.random.randint(2, 5)
    tok_class_num_labels = np.random.randint(2, 6)

    params['vocab_size'] = vocab_size
    params['max_len'] = max_len
    params['num_segments'] = num_segments
    params['d_model'] = d_model
    params['d_out'] = d_out

    params['use_attn_mask'] = use_attn_mask
    params['use_pad_mask'] = use_pad_mask

    params['use_pooler'] = use_pooler
    params['use_masked_lm'] = use_masked_lm
    params['use_next_sp'] = use_next_sp

    params['do_seq_class_task'] = do_seq_class_task
    params['do_tok_class_task'] = do_tok_class_task
    params['do_qa_task'] = do_qa_task
    params['do_mult_choice_task'] = do_mult_choice_task

    params['task_num_choices'] = task_num_choices
    params['seq_class_num_labels'] = seq_class_num_labels
    params['tok_class_num_labels'] = tok_class_num_labels

    # tf.enable_eager_execution()
    model = TransformerEncoder(**params)

    cur_len = np.random.randint(3, max_len + 1)
    if do_mult_choice_task:
        batch_size = np.random.randint(2, 4) * task_num_choices
    else:
        batch_size = np.random.randint(2, 10)

    expected_inputs = []
    expected_outputs_shape = []

    # tokens, segment_ids, pos_ids, attn_mask, pad_mask

    token_data = np.random.randint(0, vocab_size, size=(batch_size, cur_len))
    token = tf.constant(token_data, dtype=tf.int32)
    expected_inputs.append(token)
    if num_segments > 0:
        segment_ids = tf.constant(
            np.random.randint(0, num_segments, size=(batch_size, cur_len)),
            dtype=tf.int32
        )
        expected_inputs.append(segment_ids)

    pos_ids = tf.constant(
        np.array(
            [list(range(cur_len)) for _ in range(batch_size)],
            dtype=np.int32
        ),
        dtype=tf.int32
    )
    expected_inputs.append(pos_ids)

    if use_attn_mask:
        attn_mask = np.equal(token_data, 0, dtype=np.float32)
        attn_mask = attn_mask.astype(np.float32)
        attn_mask = attn_mask.reshape((batch_size, 1, cur_len))
        attn_mask = np.ones((batch_size, cur_len, 1),
                            dtype=np.float32) * attn_mask
        attn_mask = attn_mask.reshape((batch_size, 1, cur_len, cur_len))
        attn_mask = tf.constant(attn_mask)
        expected_inputs.append(attn_mask)

    if use_pad_mask:
        pad_mask = np.equal(token_data, 0, dtype=np.float32)
        pad_mask = pad_mask.astype(np.float32)
        pad_mask = pad_mask.reshape((batch_size, cur_len, 1))
        pad_mask = tf.constant(pad_mask)
        expected_inputs.append(pad_mask)

    expected_outputs_shape.append([batch_size, cur_len, d_out])
    if use_pooler:
        expected_outputs_shape.append([batch_size, d_out])
    if use_masked_lm:
        expected_outputs_shape.append([batch_size, cur_len, vocab_size])
    if use_next_sp:
        expected_outputs_shape.append([batch_size, 2])

    if do_seq_class_task:
        expected_outputs_shape.append([batch_size, seq_class_num_labels])
    if do_mult_choice_task:
        expected_outputs_shape.append(
            [batch_size // task_num_choices, task_num_choices])
    if do_tok_class_task:
        expected_outputs_shape.append(
            [batch_size, cur_len, tok_class_num_labels])
    if do_qa_task:
        expected_outputs_shape.append([batch_size, cur_len])
        expected_outputs_shape.append([batch_size, cur_len])

    if not tf.executing_eagerly():
        graph = model(expected_inputs)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            outputs = sess.run(graph)
    else:
        outputs = model(expected_inputs)

    if isinstance(outputs, (list, tuple)):
        assert len(expected_outputs_shape) == len(outputs)
        for i, v in enumerate(outputs):
            assert list(v.shape) == expected_outputs_shape[i]

    else:
        assert len(expected_outputs_shape) == 1
        assert list(outputs.shape) == expected_outputs_shape[0]
