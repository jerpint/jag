import gzip
import json

import io
import numpy as np


class FasttextEmbedding:
    """Fasttext class to get word embeddings.

    Extended description of function.

    Args:
        fname (str): name of the file containing fasttext tokens
        max_tokens (int): Max number of tokens to load in memory

    """
    def __init__(self, fname, max_tokens=1e4):

        self.fname = fname
        self.max_tokens = max_tokens
        self.load_fasttext_embeddings()

    def load_fasttext_embeddings(self):
        '''
        Loads fasttext word embeddings in memory
        '''
        filein = io.open(self.fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for count, line in enumerate(filein):
            tokens = line.rstrip().split(' ')
            data[tokens[0].lower()] = np.asarray(tokens[1:]).astype('float32')

            if count > self.max_tokens:
                break
        self.embeddings = data

    def embed(self, word):

        word = word.lower()

        if word in self.embeddings.keys():
            return self.embeddings[word]
        else:
            return np.random.random(300)


def dataset_generator(fname='data/SQuAD.jsonl.gz', batch_size=4):
    """Simple dataset generator for MRQA.

    Extended description of function.

    Args:
        fname (str): File to be processed

    Returns:
        obj: Object wtih fields defined by MRQA

    """

    # Load fasttext embeddings
    ftext = FasttextEmbedding(fname='data/wiki-news-300d-1M.vec')

    with gzip.open(fname, 'rb') as f:

        batch_context_embeddings = []
        batch_context_positions = []
        batch_questions = []

        for i, line in enumerate(f):
            instance = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in instance:
                count = 0
                continue

            else:
                if count < batch_size:

                    context_tokens = instance['context_tokens']
                    context_embeddings = []
                    context_positions = []
                    questions = instance['qas']

                    for token, pos in context_tokens:
                        context_embeddings.append(ftext.embed(token))
                        context_positions.append(pos)

                    context_embeddings = np.asarray(context_embeddings)
                    context_embeddings = np.expand_dims(context_embeddings, axis=1)

                    batch_context_embeddings.append(context_embeddings)
                    batch_context_positions.append(context_positions)
                    batch_questions.append(questions)

                    count += 1
                else:

                    yield (batch_context_embeddings,
                           batch_context_positions,
                           batch_questions)

                    batch_context_embeddings = []
                    batch_context_positions = []
                    batch_questions = []

                    count = 0
