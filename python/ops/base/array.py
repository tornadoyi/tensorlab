import tensorflow as tf


def flatten(tensor, column=False, name=None):
    if column:
        return tf.reshape(tensor, (-1, 1), name)
    else:
        return tf.reshape(tensor, (-1,), name)



def fill(dims, value, dtype=None, name=None):
    if dtype is not None: value = tf.cast(value, dtype)
    return tf.fill(dims, value, name)



def ndims(v): return tf.shape(tf.shape(v))[0]

def dim(v, i): return tf.shape(v)[i]

def dims(v):
    n = v.shape.ndims
    d = []
    for i in xrange(n): d.append(dim(v, i))
    return d


def len(v): return dim(v, 0)

def unpack(v, count):
    n = v.shape.ndims
    vs = []
    for i in xrange(count): vs.append(v[i])
    return vs

