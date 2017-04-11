import tensorflow as tf


def create(y, x, dtype=None):
    if dtype is not None:
        y, x = tf.cast(y, dtype), tf.cast(x, dtype)

    assert y.shape.ndims == x.shape.ndims
    ndims = y.shape.ndims
    assert ndims == 0 or ndims == 1

    if ndims == 0:
        return tf.convert_to_tensor([y, x])
    elif ndims == 1:
        y = tf.expand_dims(y, 1)
        x = tf.expand_dims(x, 1)
        return tf.concat([y, x], 1)

    else:
        raise Exception("Invalid points dims ", ndims)



def y(p): return p[0] if p.shape.ndims == 1 else p[:,0]

def x(p): return p[1] if p.shape.ndims == 1 else p[:,1]


def length_square(p): return x(p) ** 2 + y(p) ** 2

def length(p): return tf.sqrt(length_square(p))