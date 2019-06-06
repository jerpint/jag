import tensorflow as tf
import copy
import six
import collections
import re

from .transformer import TransformerEncoder
from . import load_bert as pretrainedModels
from . import optimization


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
    Raises:
    ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""

    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def create_model(
        config, is_training, input_ids, input_mask, segment_ids,
        load_openAI=False, load_Bert=False, useQATask=True,
        num_class_per_token=None, path=None):
    """Method for creating a model
    Args:
        config (Dict): dictionanry containing the params of the model
        is_training (bool): flag indicating training mode
        input_ids: Tensor of shape [batch_size, seq_length].
        input_mask: Tensor of shape [batch_size, seq_length].
        segment_ids: Tensor of shape [batch_size, seq_length].
        load_openAI: flag indicating whether to load OpenAI pretrained model
        load_Bert: flag indicating whether to load Google pretrained model
        useQATask: flag indicating whether to model classic QA task.
                   if False, we model instead token based classification task
        num_class_per_token (int): num of classes for eac token if `useQATask`
                   is `False`
        path (str): path to the pretrained model.
    Returns:
        the prediction of the built model for the specified inputs
    """

    config = copy.deepcopy(config)

    assert not (load_openAI and load_Bert)
    if not useQATask:
        assert (num_class_per_token is not None)

    if (load_openAI or load_Bert):
        assert (path is not None)

    # config['is_training'] = is_training

    config['use_masked_lm'] = False
    config['use_next_sp'] = False
    config['do_seq_class_task'] = False
    config['do_mult_choice_task'] = False

    config['use_pooler'] = True
    config['use_attn_mask'] = True
    config['use_pad_mask'] = True

    if not is_training:
        config['embedding_dropout'] = 0.0
        config['attention_dropout'] = 0.0
        config['residual_dropout'] = 0.0
        config['task_dropout'] = 0.0

    if useQATask:
        config['do_qa_task'] = True
        config['do_tok_class_task'] = False
    else:
        config['do_qa_task'] = False
        config['do_tok_class_task'] = True
        config['tok_class_num_labels'] = num_class_per_token

    model = None

    if load_openAI:
        model = pretrainedModels.load_openai_transformer(
            path=path, **config
        )
    elif load_Bert:
        model = pretrainedModels.load_google_bert(
            base_location=path, **config
        )
    else:
        model = TransformerEncoder(**config)

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
        input_mask = tf.ones(
            shape=[batch_size, seq_length], dtype=tf.int32
        )
    attn_mask = create_attention_mask_from_input_mask(
        input_ids, input_mask
    )
    attn_mask = tf.expand_dims(attn_mask, axis=1)

    pad_mask = tf.cast(
        tf.expand_dims(input_mask, axis=2),
        tf.float32
    )

    if segment_ids is None:
        segment_ids = tf.zeros(
            shape=[batch_size, seq_length], dtype=tf.int32
        )

    pos_ids = tf.range(seq_length, dtype=tf.int32)
    pos_ids = tf.reshape(pos_ids, [1, seq_length])
    pos_ids = tf.broadcast_to(pos_ids, [batch_size, seq_length])

    outputs = model(
        [input_ids, segment_ids, pos_ids, attn_mask, pad_mask]
    )

    if useQATask:
        _, _, start_logits, end_logits = outputs
        return (start_logits, end_logits)
    else:
        _, _, token_outputs = outputs
        return token_outputs


def model_fn_builder(
        config, init_checkpoint, learning_rate, num_train_steps,
        num_warmup_steps, use_tpu=False,
        load_openAI=False, load_Bert=False, useQATask=True,
        num_class_per_token=None, path=None, ignore_class=None):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(
                "  name = {}, shape = {}".format(name, features[name].shape)
            )

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        params = copy.deepcopy(params) if params is not None else {}
        params.update(config)

        outputs = create_model(
            config=params,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            load_openAI=load_openAI,
            load_Bert=load_Bert,
            useQATask=useQATask,
            num_class_per_token=num_class_per_token,
            path=path
        )

        if useQATask:
            start_logits, end_logits = outputs
        else:
            token_outputs = outputs

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            init_data = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint
            )
            assignment_map, initialized_variable_names = init_data

            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(
                "  name = {}, shape = {}{}".format(
                    var.name, var.shape, init_string)
            )

        output_spec = None
        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            seq_length = get_shape_list(input_ids)[1]

            def compute_loss_qa(logits, positions):
                weights = tf.where(
                    tf.equal(positions, 0),
                    tf.zeros_like(positions, dtype=logits.dtype),
                    tf.ones_like(positions, dtype=logits.dtype)
                )
                reduceWeightSum = tf.reduce_sum(
                    weights, axis=-1, keepdims=False
                )
                reduceWeightSum = tf.maximum(reduceWeightSum, 1)
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32, axis=-1)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                log_probs = tf.expand_dims(log_probs, 1)

                product_logs = (one_hot_positions * log_probs) * \
                    tf.expand_dims(weights, 2)

                product_logs = tf.reduce_sum(
                    tf.reduce_sum(product_logs, axis=-1),
                    axis=-1
                )
                product_logs = product_logs / reduceWeightSum

                loss = -tf.reduce_mean(product_logs)
                return loss

            def compute_loss_token(logits, positions):
                weights = 1.0
                if ignore_class is not None:
                    weights = tf.where(
                        tf.equal(positions, ignore_class),
                        tf.zeros_like(positions, dtype=logits.dtype),
                        tf.ones_like(positions, dtype=logits.dtype)
                    )
                losses = tf.losses.sparse_softmax_cross_entropy(
                    positions,
                    logits,
                    weights=weights,
                    reduction=None
                )
                loss = tf.reduce_mean(tf.reduce_sum(losses, axis=-1))
                return loss

            if useQATask:
                start_positions = features["start_positions"]
                end_positions = features["end_positions"]

                start_loss = compute_loss_qa(start_logits, start_positions)
                end_loss = compute_loss_qa(end_logits, end_positions)

                total_loss = (start_loss + end_loss) / 2.0
            else:
                token_classes = features["token_classes"]
                total_loss = compute_loss_token(token_outputs, token_classes)

            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps,
                    num_warmup_steps, use_tpu
                )

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn
                )
            else:
                eval_metrics = None
                if not useQATask:
                    def metric_fn(labels, logits):
                        weights = None
                        if ignore_class is not None:
                            weights = tf.where(
                                tf.equal(labels, ignore_class),
                                tf.zeros_like(labels, dtype=logits.dtype),
                                tf.ones_like(labels, dtype=logits.dtype)
                            )
                        accuracy = tf.metrics.accuracy(
                            labels=labels,
                            predictions=tf.argmax(logits, axis=2),
                            weights=weights
                        )
                        return {"accuracy": accuracy}

                    eval_metrics = (metric_fn, [token_classes, token_outputs])

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn
                )

        elif mode == tf.estimator.ModeKeys.PREDICT:
            if useQATask:
                predictions = {
                    "unique_ids": unique_ids,
                    "start_logits": start_logits,
                    "end_logits": end_logits,
                }
            else:
                predictions = {
                    "unique_ids": unique_ids,
                    "token_logits": token_outputs,
                }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn
            )
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn
