import tensorflow as tf
import tensorflow_hub as hub

# Other modules
# "https://tfhub.dev/google/universal-sentence-encoder-large/3"
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Import the Universal Sentence Encoder's TF Hub module
embed_model = hub.Module(module_url)
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:

    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embed_model(messages))
