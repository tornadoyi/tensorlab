import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


def batch_normalization(
        x,
        is_training,
        use_bias = False,
        epsilon = 0.001,
        moving_average_decay = 0.9997,
        init_moving_mean = None,
        init_moving_variance = None,
        ):

    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias:
        bias = tf.Variable(np.zeros(params_shape))
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = tf.Variable(np.zeros(params_shape))
    gamma = tf.Variable(np.ones(params_shape))

    if init_moving_mean is None: init_moving_mean = np.zeros(params_shape)
    moving_mean = tf.Variable(init_moving_mean, trainable=False)

    if init_moving_variance is None: init_moving_variance = np.ones(params_shape)
    moving_variance = tf.Variable(init_moving_variance, trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, moving_average_decay)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, moving_average_decay)

    #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
    #x.set_shape(inputs.get_shape()) ??

    return x, [update_moving_mean, update_moving_variance]