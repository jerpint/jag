import re
import numpy as np
from collections import Counter
from tensorflow.keras.utils import to_categorical


def load_text(filename: str) -> str:
    """Load text into memory.

    Args:
        filename(str): text file.

    Returns:
        text(str): raw text.
    """
    with open(filename, 'r') as f:
        text = f.read()

    return text


def clean_text(raw_text: str):
    """Clean raw text.

    Args:
        raw_text(str): text.

    Returns:
        cleaned_text(str): cleaned_text
    """
    tokens = raw_text.split()
    cleaned_text = '_'.join(tokens)
    pattern = re.compile(r'（.+?）')
    cleaned_text = pattern.sub('', cleaned_text)

    return cleaned_text


def save_text(lines: str, filename: str) -> None:
    """Save text line by line.

    Args:
        lines(list): texts.
        filename(str): text file.
    """
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def create_vocabulary(text: str, vocab_size=100000):
    """Create vocabulary dictionaries.

    Args:
        text(str): raw text.

    Returns:
        token2id(dict): token to id mapping.
        id2token(dict): id to token mapping.
    """
    token2id = {'<PAD>': 0, '<OOV>': 1}
    id2token = {0: '<PAD>', 1: '<OOV>'}
    freq = Counter(text)
    for token, _ in freq.most_common(vocab_size):
        tok_id = len(token2id)
        token2id[token] = tok_id
        id2token[tok_id] = token

    return token2id, id2token


def create_dataset(text, token2id, maxlen=10):
    """Create a dataset.

    Args:
        text(str): text.
        token2id(dict): token to id mapping.
        maxlen(int): max sequence length.

    Returns:
        X(ndarray): encoded token sequences.
        y(ndarray): encoded label sequences.
    """
    sequences = []
    for i in range(maxlen, len(text)):
        seq = text[i - maxlen: i + 1]
        encoded = [token2id.get(token, 1) for token in seq]
        sequences.append(encoded)

    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]

    return X, y


def preprocess_dataset(X, y, vocab_size):
    X = [to_categorical(x, num_classes=vocab_size) for x in X]
    y = to_categorical(y, num_classes=vocab_size)

    return X, y
