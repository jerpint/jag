def test_dataloader():
    import tensorflow as tf
    from jag.data.data_handler import DatasetHandler
    from jag.data.feature_handler import convert_examples_to_features
    from jag.utils.bert_tokenizer import BertTokenizer
    from jag.data.dataloader import FeatureWriter, input_fn_builder

    max_seq_length = 256
    ctx_stride = 128
    max_query_length = 64
    max_answer_num = 3
    is_training = True
    drop_remainder = False

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased',
        do_lower_case=False,
        cache_dir='./unit_tests/data/'
    )
    data_hnd = DatasetHandler(
        data_src='./jag/data/mrqa_urls_sample.txt',
        cache_dir='./unit_tests/data/'
    )
    examples = data_hnd.read_mrqa_examples(is_training=is_training)
    data_writer = FeatureWriter(
        './unit_tests/data/train_data.TFRecord',
        is_training=is_training
    )

    convert_examples_to_features(
        examples, tokenizer, max_seq_length=max_seq_length,
        ctx_stride=ctx_stride, max_query_length=max_query_length,
        max_answer_num=max_answer_num,
        output_fn=data_writer.process_feature,
        keep_partial_answer_span=False,
        same_token_class_per_answer_token=True,
        unique_id_start=1000000000
    )
    data_writer.close()

    train_input_fn = input_fn_builder(
        input_file=data_writer.filename,
        seq_length=max_seq_length,
        max_answer_num=max_answer_num,
        is_training=is_training,
        drop_remainder=drop_remainder,
    )

    dataset = train_input_fn({'batch_size': 8})
    feature = tf.data.experimental.get_single_element(dataset)

    assert feature['input_ids'].shape[1] == max_seq_length
    assert feature['input_mask'].shape[1] == max_seq_length
    assert feature['segment_ids'].shape[1] == max_seq_length
