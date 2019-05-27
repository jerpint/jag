import io
import numpy as np


class FasttextEmbedding:
    """Fasttext class to get word embeddingsline.

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
        filein = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for count, line in enumerate(filein):
            tokens = line.rstrip().split(' ')
            data[tokens[0].lower()] = np.asarray(tokens[1:]).astype('float32')

            if count > self.max_tokens:
                break
        self.embeddings = data

    def get_word(self, word):

        word = word.lower()

        if word in self.embeddings.keys():
            return self.embeddings[word]
        else:
            print("Random")
            return np.random.random(300)


if __name__ == "__main__":

    fname = 'data/wiki-news-300d-1M.vec'
    embeddings = FasttextEmbedding(fname)
    print(embeddings.get_word('cool'))
