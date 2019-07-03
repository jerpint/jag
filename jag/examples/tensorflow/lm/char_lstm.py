# -*- coding: utf-8 -*-
"""
Character based language model.
"""
import jag.examples.tensorflow.lm.preprocess as lm_preprocess
import mlflow
import jag.examples.tensorflow.lm.mlflow_utils as mlflow_utils
import tensorflow as tf
import tensorflow.logging
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

import argparse


def generate_text(model, char2id, id2char, seed_text, maxlen=10, iter=20):
    """Generate a sequence of characters.

    Args:
        model: trained model.
        char2id(dict): character to id mapping.
        id2char(dict): id to character mapping.
        maxlen: max sequence length.
        seed_text: seed text for generating new text.
        iter: number of iteration to generate character.

    Returns:
        text(str): generated text.
    """
    encoded = [char2id[char] for char in seed_text]
    for _ in range(iter):
        x = pad_sequences([encoded], maxlen=maxlen, truncating='pre')
        y = model.predict_classes(x, verbose=0)
        encoded.append(y[0])
    decoded = [id2char[c] for c in encoded]
    text = ''.join(decoded)

    return text


def create_model(vocab_size, embedding_dim=50):
    """Create a model.

    Args:
        vocab_size(int): vocabulary size.
        embedding_dim(int): embedding dimension.

    Returns:
        model: model object.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))
    model.add(LSTM(75))
    model.add(Dense(vocab_size, activation='softmax'))

    return model


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='data/1661-0.txt',
                        help="echo the string you use here")
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    raw_text = lm_preprocess.load_text(args.data)
    cleaned_text = lm_preprocess.clean_text(raw_text)
    print(cleaned_text)
    cleaned_text = cleaned_text[:10000]

    char2id, id2char = lm_preprocess.create_vocabulary(cleaned_text)
    vocab_size = len(char2id)
    tf.logging.info('Vocabulary size: {}'.format(vocab_size))

    X, y = lm_preprocess.create_dataset(cleaned_text, char2id)

    model = create_model(vocab_size)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    with mlflow.start_run() as run:  # noqa
        model.fit(X, y, epochs=100, verbose=2, callbacks=[mlflow_utils.MLflowLogger()])


if __name__ == '__main__':
    tf.app.run()
