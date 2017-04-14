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

def clip(p, y_range, x_range):
    y_min, y_max = tf.split(y_range, [1, 1])
    x_min, x_max = tf.split(x_range, [1, 1])
    return create(
        tf.clip_by_value(y(p), y_min, y_max),
        tf.clip_by_value(x(p), x_min, x_max),
    )


def clip_y(p, min, max):
    return create(
        tf.clip_by_value(y(p), min, max),
        x(p)
    )

def clip_x(p, min, max):
    return create(
        y(p),
        tf.clip_by_value(x(p), min, max)
    )


def length_square(p): return x(p) ** 2 + y(p) ** 2

def length(p): return tf.sqrt(length_square(p))