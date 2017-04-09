import tensorflow as tf


def create(y, x, dtype=None):
    if dtype is None:
        return tf.convert_to_tensor([y, x], dtype)
    else:
        return tf.cast([y, x], dtype)

def y(p): return p[0] if p.shape.ndims == 1 else p[:,0]

def x(p): return p[1] if p.shape.ndims == 1 else p[:,1]
