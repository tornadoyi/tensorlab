import tensorflow as tf
import numpy as np



def shape(t):
    if isinstance(t, tuple) or isinstance(t, list):
        return t

    elif isinstance(t, np.ndarray):
        return t.shape

    elif isinstance(t, tf.Tensor) or \
         isinstance(t, tf.Variable) or \
         isinstance(t, tf.Constant):

        tensor_shape =  t.get_shape()
        return tuple([d.value for d in tensor_shape])

    else:
        raise Exception("type of t must be tuple, list, np.ndarray, tf.Tensor")
