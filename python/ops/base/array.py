import tensorflow as tf


def flatten(tensor, column=False):
    if column:
        return tf.reshape(tensor, (-1, 1))
    else:
        return tf.reshape(tensor, (-1,))



def fill(dims, value, dtype=None, name=None):
    if dtype is not None: value = tf.cast(value, dtype)
    return tf.fill(dims, value, name)