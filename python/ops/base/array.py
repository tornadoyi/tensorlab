import tensorflow as tf


def flatten(tensor, column=False, name=None):
    if column:
        return tf.reshape(tensor, (-1, 1), name)
    else:
        return tf.reshape(tensor, (-1,), name)



def fill(dims, value, dtype=None, name=None):
    if dtype is not None: value = tf.cast(value, dtype)
    return tf.fill(dims, value, name)



def dims(v): return tf.shape(tf.shape(v))[0]

def dim(v, i): return tf.shape(v)[i]


def len(v): return dim(v, 0)