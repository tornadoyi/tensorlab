import tensorflow as tf

epsilon = 2.2204460492503131e-16

def check_tensor(*args):
    list = []
    for v in args:
        v = v if type(v) == tf.Tensor else tf.convert_to_tensor(v)
        list.append(v)

    return list[0] if len(list) == 1 else list