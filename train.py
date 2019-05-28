import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from data.data import dataset_generator


# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed_model = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

data = dataset_generator()

context_embeddings, context_positions, questions = next(data)


# TODO: This is sample data from batch for debugging, fix
context_embeddings = context_embeddings[0]
batch_size = context_embeddings.shape[1]
time_steps = context_embeddings.shape[0]
num_features = context_embeddings.shape[2]
raw_question = questions[0][0]['question']

words_in_context = tf.placeholder(tf.float32, [time_steps+1, batch_size, num_features])

lstm = tf.keras.layers.LSTM(units=2)
lstm_forward = lstm(words_in_context)

question_embedding_placeholder = tf.placeholder(tf.float32, [batch_size, 512])
dense_linear = tf.keras.layers.Dense(units=300)
dense = dense_linear(question_embedding_placeholder)

with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    question_embedding = sess.run(embed_model([raw_question]))
    reshaped_embedding = sess.run(dense, {question_embedding_placeholder: question_embedding})
    reshaped_embedding = np.expand_dims(reshaped_embedding, axis=0)

    context_embeddings = np.concatenate((reshaped_embedding, context_embeddings))
    context_pred = sess.run(lstm_forward, {words_in_context: context_embeddings})
