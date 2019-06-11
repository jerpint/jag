import tensorflow as tf
import collections


class FeatureWriter(object):
    """A class utility for writing InputFeature to TF example file.
        Args:
            filename (str): path to a file in which the feature data will be saved.
            is_training (bool): Flag indicating whether or not the data to be
                saved are for training purposes.
    """

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
    """ Creates an `input_fn` closure tthat defines our dataset
        from a TFrecord data file. be passed to our model.
        Args:
            input_file (str): path to a TFRecord data file from which the
                feature data will be read.
            seq_length (int): The maximum length of input data.
            max_answer_num (int): the maximum number of answer per question
            is_training (bool): flag indicating whether or not the data to be
                saved are for training purposes.
            drop_remainder (bool): whether to drop the remaining data if the
                len of the dataset is not a multiple of the batch size.
        Returns:
            a closure function that will instantiate a dataset based on the params
            received as arguments.

    """

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
                t = tf.cast(t, tf.int32)  # tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function.

            params (Dict): parameters of the dataset. Must contains as least the
                'batch_size' key.

        """
        assert "batch_size" in params

        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            )
        )

        return d

    return input_fn


if __name__ == '__main__':
    from jag.data.data_handler import DatasetHandler
    from jag.data.feature_handler import convert_examples_to_features
    from jag.utils.bert_tokenizer import BertTokenizer

    max_seq_length = 256
    ctx_stride = 128
    max_query_length = 64
    max_answer_num = 3
    is_training = True
    drop_remainder = False

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased',
        do_lower_case=False,
        cache_dir='./cache/tokenizer/'
    )
    data_hnd = DatasetHandler(
        data_src='./cache/mrqa_urls.txt', cache_dir='./cache/')
    examples = data_hnd.read_mrqa_examples(is_training=is_training)
    data_writer = FeatureWriter(
        './cache/train_data.TFRecord',
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
    # dataset_size = tf.data.experimental.cardinality(dataset)

    print('Data writer size: ', data_writer.num_features)

    feature = tf.data.experimental.get_single_element(dataset)
    print('element:\n', feature)

    assert feature['input_ids'].shape[1] == max_seq_length
    assert feature['input_mask'].shape[1] == max_seq_length
    assert feature['segment_ids'].shape[1] == max_seq_length
