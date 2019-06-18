import numpy as np
import jag.examples.tensorflow.lm.preprocess as lm_preprocess


def test_create_vocabulary():
    input_text = 'this is a test for testing a vocab'
    exp_token2id = {'<PAD>': 0, '<OOV>': 1, ' ': 2, 't': 3, 's': 4, 'i': 5, 'a': 6}
    exp_id2token = {0: '<PAD>', 1: '<OOV>', 2: ' ', 3: 't', 4: 's', 5: 'i', 6: 'a'}

    token2id, id2token = lm_preprocess.create_vocabulary(
        input_text, len(exp_token2id)-2)

    assert exp_token2id == token2id
    assert exp_id2token == id2token


def test_create_dataset_basic():
    input_text = 'abcd dbac'
    token2id = {'<PAD>': 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}

    exp_X = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 1], [5, 1, 5], [1, 5, 3], [5, 3, 2]])
    exp_y = np.array([5, 1, 5, 3, 2, 4])

    X, y = lm_preprocess.create_dataset(input_text, token2id, maxlen=3)
    assert np.array_equal(X, exp_X)
    assert np.array_equal(y, exp_y)


def test_create_dataset_oov():
    input_text = 'abzd!a'
    token2id = {'<PAD>': 0, '<OOV>': 1, ' ': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6}

    exp_X = np.array([[3, 4, 1], [4, 1, 6], [1, 6, 1]])
    exp_y = np.array([6, 1, 3])

    X, y = lm_preprocess.create_dataset(input_text, token2id, maxlen=3)
    assert np.array_equal(X, exp_X)
    assert np.array_equal(y, exp_y)
