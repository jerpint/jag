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


question_embedding_use = embed_model([raw_question])
dense_linear = tf.keras.layers.Dense(units=num_features)
question_embedding = dense_linear(question_embedding_use)
question_embedding = tf.reshape(question_embedding, [batch_size, 1, num_features])

context_embedding = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])

embedded_question_context = tf.concat([question_embedding, context_embedding], axis=0)

lstm = tf.keras.layers.LSTM(units=2)
lstm_forward = lstm(embedded_question_context)

with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    context_pred = sess.run(lstm_forward, {context_embedding: context_embeddings})
