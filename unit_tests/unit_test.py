def test_tensorflow():
    '''Basic test to make sure tensorflow is properly installed'''

    import tensorflow as tf

    a = tf.constant(3.0)
    b = tf.constant(4.0)
    graph = a + b

    with tf.Session() as sess:
        out = sess.run(graph)

    assert out == 7
