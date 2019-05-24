import io
import numpy as np


class FasttextEmbedding:

    def __init__(self, fname, max_tokens=1e4):

        self.fname = fname
        self.max_tokens = max_tokens
        self.load_fasttext_embeddings()

    def load_fasttext_embeddings(self):
        '''
        Loads fasttext word embeddings in memory


        '''
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for count, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            #  data[tokens[0].lower()] = map(float, tokens[1:])
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
