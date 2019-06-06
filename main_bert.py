import tensorflow as tf
import argparse
import os
import json
import random

from data import bert_tokenization as tokenization
from data import data_handler as data_hnd
from models import model_handler as model_hnd
from eval import eval_handler as eval_hnd


def manage_arguments():
    parser = argparse.ArgumentParser('MRQA BERT')

    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument(
        '--load_openAI', default=False, action='store_true',
        help='Whether to load pretrained weigth of OpenAI.'
    )
    parser.add_argument(
        '--load_Bert', default=False, action='store_true',
        help='Whether to load pretrained weigth of BERT.'
    )
    parser.add_argument(
        '--pretrained_path', type=str, default=None,
        help='path to load a pretrained model'
    )
    parser.add_argument(
        '--no_qa_task', default=False, action='store_true',
        help='Whether to load pretrained weigth of OpenAI.'
    )
    parser.add_argument(
        '--keep_partial_answer_span', default=False, action='store_true',
        help='Keeps partial anwer span in a chunk'
    )
    parser.add_argument(
        '--same_token_class_per_answer_token', default=False, action='store_true',
        help="Use the same class for all answer's tokens"
    )
    parser.add_argument(
        '--seed', type=int, default=12345,
        help='the seed for randomness'
    )
    parser.add_argument(
        '--model_config_file', type=str, required=True,
        help="The config json file corresponding to the transformer model. " +
        "This specifies the model architecture."
    )
    parser.add_argument(
        '--vocab_file', type=str, default=None,
        help='The vocabulary file that to be used'
    )
    parser.add_argument(
        '--pretrained_model_vocab', type=str, default=None,
        help='if specified, we used the vocab associated with the pretrained model.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./outputs',
        help='The output directory where the model checkpoints will be written'
    )
    parser.add_argument(
        '--train_file', type=str, default=None,
        help='the file containing the training datasets to use'
    )
    parser.add_argument(
        '--eval_file', type=str, default=None,
        help='the file containing the datasets for eval purposes'
    )
    parser.add_argument(
        '--predict_file', type=str, default=None,
        help='the file containing the datasets for test purposes'
    )
    parser.add_argument(
        '--init_checkpoint', type=str, default=None,
        help='Initial checkpoint (usually from a pre-trained BERT model).'
    )
    parser.add_argument(
        '--do_lower_case', default=False, action='store_true',
        help='Whether to lower case the input text. Should be True for uncased' +
        ' models and False for cased models.'
    )
    parser.add_argument(
        '--max_seq_length', type=int, default=384,
        help='The maximum total input sequence length after WordPiece tokenization.'
    )
    parser.add_argument(
        '--ctx_stride', type=int, default=128,
        help='When splitting up a long document into chunks, how much stride to' +
        'take between chunks.'
    )
    parser.add_argument(
        '--max_query_length', type=int, default=64,
        help="The maximum number of tokens for the question. Questions longer than " +
        "this will be truncated to this length."
    )
    parser.add_argument(
        '--do_train', default=False, action='store_true',
        help='Whether to run training.'
    )
    parser.add_argument(
        '--do_eval', default=False, action='store_true',
        help='Whether to run eval on the dev set.'
    )
    parser.add_argument(
        '--do_predict', default=False, action='store_true',
        help='Whether to run predict on the dev set.'
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=8,
        help='Total batch size for training.'
    )
    parser.add_argument(
        '--predict_batch_size', type=int, default=8,
        help='Total batch size for predictions.'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=8,
        help='Total batch size for evaluation.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=5e-5,
        help='The initial learning rate for Adam.'
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=3,
        help='Total number of training epochs to perform.'
    )
    parser.add_argument(
        '--warmup_proportion', type=float, default=0.1,
        help="Proportion of training to perform linear learning rate warmup for. " +
        "E.g., 0.1 = 10%% of training."
    )
    parser.add_argument(
        '--save_checkpoints_steps', type=int, default=1000,
        help="How often to save the model checkpoint."
    )
    parser.add_argument(
        '--iterations_per_loop', type=int, default=1000,
        help="How many steps to make in each estimator call."
    )
    parser.add_argument(
        '--n_best_size', type=int, default=20,
        help="The total number of n-best predictions to generate in the " +
        "nbest_predictions.json output file."
    )
    parser.add_argument(
        '--max_answer_length', type=int, default=30,
        help="The maximum length of an answer that can be generated. This is needed " +
        "because the start and end predictions are not conditioned on one another."
    )
    parser.add_argument(
        '--max_answer_num', type=int, default=1,
        help="The maximum number of answers per question"
    )
    parser.add_argument(
        '--use_tpu', default=False, action='store_true',
        help='Whether to use TPU or GPU/CPU.'
    )
    parser.add_argument(
        '--tpu_name', type=str, default=None,
        help="The Cloud TPU to use for training. This should be either the name " +
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 " +
        "url."
    )
    parser.add_argument(
        '--tpu_zone', type=str, default=None,
        help="[Optional] GCE zone where the Cloud TPU is located in. If not " +
        "specified, we will attempt to automatically detect the GCE project from " +
        "metadata."
    )
    parser.add_argument(
        '--gcp_project', type=str, default=None,
        help="[Optional] Project name for the Cloud TPU-enabled project. If not " +
        "specified, we will attempt to automatically detect the GCE project from " +
        "metadata."
    )
    parser.add_argument(
        '--master', type=str, default=None,
        help="[Optional] TensorFlow master URL."
    )
    parser.add_argument(
        '--num_tpu_cores', type=int, default=8,
        help='Only used if `use_tpu` is True. Total number of TPU cores to use.'
    )
    parser.add_argument(
        '--verbose_logging', default=False, action='store_true',
        help="If true, all of the warnings related to data processing will be printed. " +
        "A number of warnings are expected for a normal SQuAD evaluation."
    )
    parser.add_argument(
        '--version_2_with_negative', default=False, action='store_true',
        help="If true, somes examples may not have an answer."
    )
    parser.add_argument(
        '--null_score_diff_threshold', type=float, default=0.0,
        help='If null_score - best_non_null is greater than the threshold predict null.'
    )

    args = parser.parse_args()

    assert not (args.vocab_file and args.pretrained_model_vocab), "Both `vocab_file`" + \
        " and `pretrained_model_vocab` should not be specified together"
    assert (args.vocab_file or args.pretrained_model_vocab), "Either `vocab_file`" + \
        " or `pretrained_model_vocab` should be specified"
    assert not (args.load_openAI and args.load_Bert), "Both `load_openAI`" + \
        " and `load_Bert` should not be specified together"
    if (args.load_openAI or args.load_Bert):
        assert args.pretrained_path, "`pretrained_path` needs to be specified!"

    return args


def prepare_run_config(args):
    tpu_cluster_resolver = None
    if args.use_tpu and args.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=args.tpu_zone, project=args.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=args.master,
        model_dir=args.output_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=args.iterations_per_loop,
            num_shards=args.num_tpu_cores,
            per_host_input_for_training=is_per_host)
    )

    return run_config


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    args = manage_arguments()

    with tf.gfile.GFile(args.model_config_file, "r") as reader:
        text = reader.read()
    config = json.loads(text)

    tf.gfile.MakeDirs(args.output_dir)
    tf.gfile.MakeDirs(args.cache_dir)

    special_tokens = (
        "[UNK]", "[SEP]", "[PAD]", "[CLS]",
        "[MASK]", "[TLE]", "[DOC]", "[PAR]"
    )

    if args.vocab_file:
        tokenizer = tokenization.BertTokenizer(
            args.vocab_file, do_lower_case=args.do_lower_case,
            max_len=args.max_seq_length,
            do_basic_tokenize=True, never_split=special_tokens
        )
    else:
        tf.gfile.MakeDirs(args.cache_dir + '/tokenizer/')
        tokenizer = tokenization.BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.pretrained_model_vocab,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir + '/tokenizer/',
            max_len=args.max_seq_length,
            do_basic_tokenize=True, never_split=special_tokens
        )

    run_config = prepare_run_config(args)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if args.do_train:
        data_train_handler = data_hnd.DatasetHandler(
            cache_dir=args.cache_dir,
            data_src=args.train_file
        )
        train_examples = data_train_handler.read_mrqa_examples(True)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs
        )
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(args.seed)
        rng.shuffle(train_examples)

    model_fn = model_hnd.model_fn_builder(
        config, args.init_checkpoint, args.learning_rate, num_train_steps,
        num_warmup_steps, use_tpu=args.use_tpu,
        load_openAI=args.load_openAI, load_Bert=args.load_Bert,
        useQATask=not args.no_qa_task,
        num_class_per_token=3 if args.same_token_class_per_answer_token else 4,
        path=args.pretrained_path, ignore_class=0
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=args.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
        eval_batch_size=args.eval_batch_size
    )

    if args.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = data_hnd.FeatureWriter(
            filename=os.path.join(args.output_dir, "train.tf_record"),
            is_training=True
        )
        data_train_handler.convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            ctx_stride=args.ctx_stride,
            max_query_length=args.max_query_length,
            max_answer_num=args.max_answer_num,
            output_fn=train_writer.process_feature,
            keep_partial_answer_span=args.keep_partial_answer_span,
            same_token_class_per_answer_token=args.same_token_class_per_answer_token,
            unique_id_start=1000000000
        )
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", args.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        del train_examples

        train_input_fn = data_hnd.input_fn_builder(
            input_file=train_writer.filename,
            seq_length=args.max_seq_length,
            max_answer_num=args.max_answer_num,
            is_training=True,
            drop_remainder=True if args.use_tpu else False
        )

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if args.do_eval:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        data_eval_handler = data_hnd.DatasetHandler(
            cache_dir=args.cache_dir,
            data_src=args.eval_file
        )
        eval_examples = data_eval_handler.read_mrqa_examples(True)
        eval_writer = data_hnd.FeatureWriter(
            filename=os.path.join(args.output_dir, "eval.tf_record"),
            is_training=True
        )
        data_eval_handler.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            ctx_stride=args.ctx_stride,
            max_query_length=args.max_query_length,
            max_answer_num=args.max_answer_num,
            output_fn=eval_writer.process_feature,
            keep_partial_answer_span=args.keep_partial_answer_span,
            same_token_class_per_answer_token=args.same_token_class_per_answer_token,
            unique_id_start=1000000000
        )
        eval_writer.close()

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", eval_writer.num_features)
        tf.logging.info("  Batch size = %d", args.eval_batch_size)

        del eval_examples

        eval_steps = eval_writer.num_features // args.eval_batch_size

        eval_input_fn = data_hnd.input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=args.max_seq_length,
            max_answer_num=args.max_answer_num,
            is_training=True,
            drop_remainder=True if args.use_tpu else False
        )

        estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    if args.do_predict:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        data_pred_handler = data_hnd.DatasetHandler(
            cache_dir=args.cache_dir,
            data_src=args.predict_file
        )
        pred_examples = data_pred_handler.read_mrqa_examples(False)
        pred_writer = data_hnd.FeatureWriter(
            filename=os.path.join(args.output_dir, "pred.tf_record"),
            is_training=False
        )

        pred_features = []

        def append_feature(feature):
            pred_features.append(feature)
            pred_writer.process_feature(feature)

        data_pred_handler.convert_examples_to_features(
            examples=pred_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            ctx_stride=args.ctx_stride,
            max_query_length=args.max_query_length,
            max_answer_num=args.max_answer_num,
            output_fn=append_feature,
            keep_partial_answer_span=args.keep_partial_answer_span,
            same_token_class_per_answer_token=args.same_token_class_per_answer_token,
            unique_id_start=1000000000
        )
        pred_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(pred_examples))
        tf.logging.info("  Num split examples = %d", len(pred_features))
        tf.logging.info("  Batch size = %d", args.predict_batch_size)

        predict_input_fn = data_hnd.input_fn_builder(
            input_file=pred_writer.filename,
            seq_length=args.max_seq_length,
            max_answer_num=args.max_answer_num,
            is_training=False,
            drop_remainder=False
        )

        # If running eval on the TPU, you will need to specify the number of
        # steps.

        all_results = []
        for result in estimator.predict(predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))

            unique_id = int(result["unique_ids"])
            if not args.no_qa_task:
                start_logits = [float(x) for x in result["start_logits"].flat]
                end_logits = [float(x) for x in result["end_logits"].flat]
                all_results.append(
                    eval_hnd.RawQAResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits
                    )
                )
            else:
                token_logits = [float(x) for x in result["token_logits"].flat]
                all_results.append(
                    eval_hnd.RawTokenClassResult(
                        unique_id=unique_id,
                        token_logits=token_logits,
                    )
                )

        output_prediction_file = os.path.join(
            args.output_dir, "predictions.json"
        )
        output_nbest_file = os.path.join(
            args.output_dir, "nbest_predictions.json"
        )
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds.json"
        )

        eval_prediction_handler = eval_hnd.EvalHandler(
            version_2_with_negative=args.version_2_with_negative,
            verbose_logging=args.verbose_logging,
            null_score_diff_threshold=args.null_score_diff_threshold
        )

        if not args.no_qa_task:
            eval_prediction_handler.write_qa_predictions(
                pred_examples, pred_features, all_results,
                args.n_best_size, args.max_answer_length,
                output_prediction_file,
                output_nbest_file, output_null_log_odds_file,
                tokenizer=tokenization.BasicTokenizer(
                    do_lower_case=args.do_lower_case),
                wordpiece_indicator='##'
            )
        else:
            pass

    print('done MAIN!')


if __name__ == "__main__":
    tf.app.run(main=main)
    print('done!')
