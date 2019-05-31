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

# TODO: Fix: this is sample data from a batch for initializing sizes
context_embeddings = context_embeddings[0]
batch_size = context_embeddings.shape[0]
time_steps = context_embeddings.shape[1]
num_features = context_embeddings.shape[2]

raw_question_ph = tf.placeholder(tf.string, name='raw_queestion')
context_emb_ph = tf.placeholder(tf.float32, shape=[batch_size, None, num_features],
                                name='context_embedding')
indices_ph = tf.placeholder(tf.int32, name='label_indices', shape=[None])


def embed_question_and_context(raw_question, context_embedding):
    """Creates a single embedding from the question and context tokens.

    We use the Universal Sentence Encoder to encode the question and
    a linear mapping to reduce its dimension to match the fasttext
    dimension of our tokens.

    Args:
        raw_question (str): The full question as a single string
        context_embedding (ndarray): A tensor containing the token embeddings of shape
           [batch_size, time_steps, num_features].

    Returns:
        tf.tensor: A tf.tensor with the question concatenated at the first time step
    """
    question_embedding_use = embed_model([raw_question_ph])
    dense_linear = tf.keras.layers.Dense(units=num_features)
    question_embedding = dense_linear(question_embedding_use)
    question_embedding = tf.reshape(question_embedding, [batch_size, 1, num_features])

    return tf.concat([question_embedding, context_emb_ph], axis=1)


def vanilla_lstm(embedded_question_context):
    """A simple lstm model.

    Args:
        embedded_question_context (tf.tensor): A tf.tensor with the question concatenated
        at the first time step has shape [batch_size, time_steps, num_features].

    Returns:
        tf.tensor: A tf.tensor with the logit outputs of the LSTM
    """
    lstm = tf.keras.layers.LSTM(units=2, return_sequences=True)
    logits = lstm(embedded_question_context)

    #  First time step is the question embedding so we ignore it
    return logits[:, 1:, :]


def make_oh_labels(indices_ph):
    """Genrate one hot labels for the target.

    Args:
        indices_ph (tf.placeholder): A placeholder of the indices to words that
        correspond to the answerthe question concatenated

    Returns:
        tf.tensor: A tf.tensor with the one-hot labels
    """
    labels = tf.one_hot(indices=indices_ph, depth=2)
    labels = tf.expand_dims(labels, axis=0)

    return labels


embedded_qc = embed_question_and_context(raw_question_ph, context_emb_ph)
logits = vanilla_lstm(embedded_qc)
preds = tf.nn.softmax(logits)
labels = make_oh_labels(indices_ph)

sce_loss = tf.losses.softmax_cross_entropy(labels, logits)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(sce_loss)

with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for batch in range(10):

        print("Batch :", batch)

        context_embeddings, context_positions, questions = next(data)

        # TODO: This is only assuming batch_size=1, will need to generalize

        context_embeddings = context_embeddings[0]
        batch_size = context_embeddings.shape[0]
        time_steps = context_embeddings.shape[1]
        num_features = context_embeddings.shape[2]

        raw_question = questions[0][0]['question']

        token_spans = questions[0][0]['detected_answers'][0]['token_spans'][0]
        indices = np.zeros(time_steps, dtype=np.int32)
        indices[token_spans[0]:token_spans[1]+1] = 1

        sess.run(train_op,
                 {context_emb_ph: context_embeddings,
                  indices_ph: indices,
                  raw_question_ph: raw_question})
