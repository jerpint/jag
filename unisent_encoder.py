import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from utils.fasttext import FasttextEmbedding
from data.data import dataset_generator

# Load fasttext embeddings
ftext = FasttextEmbedding(fname='data/wiki-news-300d-1M.vec')

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed_model = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

data = dataset_generator()

obj = next(data)
context_tokens = obj['context_tokens']
context_embeddings = []
context_position = []
question = obj['qas'][0]['question']

for token, pos in context_tokens:
    context_embeddings.append(ftext.embed(token))
    context_position.append(pos)

context_embeddings = np.asarray(context_embeddings)
context_embeddings = np.expand_dims(context_embeddings, axis=1)
batch_size = context_embeddings.shape[1]
time_steps = context_embeddings.shape[0]
num_features = context_embeddings.shape[2]


words_in_context = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
lstm = tf.keras.layers.LSTM(units=3)
lstm_forward = lstm(words_in_context)

with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    question_embedding = sess.run(embed_model([question]))
    context_pred = sess.run(lstm_forward, {words_in_context: context_embeddings})
