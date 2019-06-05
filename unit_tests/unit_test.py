import tensorflow as tf
import numpy as np

from data.data import dataset_generator


def test_tensorflow():
    '''Basic test to make sure tensorflow is properly installed'''

    a = tf.constant(3.0)
    b = tf.constant(4.0)
    graph = a + b

    with tf.Session() as sess:
        out = sess.run(graph)

    assert out == 7


def test_generator():

    batch_size = 2

    data = dataset_generator(batch_size=batch_size)
    context_embeddings, context_positions, questions = next(data)

    assert len(context_embeddings) == batch_size
    assert len(context_positions) == batch_size
    assert len(questions) == batch_size

    for ce in context_embeddings:
        assert type(ce) == np.ndarray

    for cp in context_positions:
        assert type(cp) == list

    for qs in questions:
        keys = ['answers', 'question', 'id', 'qid', 'question_tokens', 'detected_answers']
        assert (keys == list(qs[0].keys()))
