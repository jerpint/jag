import argparse
import sentencepiece as spm

"""
Script to launch the unsupervised text tokenizer without compiling from sources.
Source: https://github.com/google/sentencepiece

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Script to launch unsupervised text tokenizer')
    parser.add_argument('input', type=str, default='rawdata.txt',
                        help='one-sentence-per-line raw corpus file.')
    parser.add_argument('model-prefix', type=str, default='m',
                        help='output model name prefix.' +
                        '<model_name>.model and <model_name>.vocab are generated.')
    parser.add_argument('vocab-size', type=int, default=8000,
                        help='vocabulary size, e.g., 8000, 16000, or 32000.')

    args = parser.parse_args()
    print('Generating the vocabulary with SentencePiece')
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={}'.format(
            args.input, args.model_prefix, args.vocab_size))
