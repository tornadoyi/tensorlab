import os
import tensorflow as tf



def variable_scope_join(*args): return os.path.join(*args).replace("\\", "/")


def empty_variable_scope(name, key=tf.GraphKeys.GLOBAL_VARIABLES):
    i = 0
    while True:
        sub_name = name + "_" + str(i)
        scope_name = variable_scope_join(tf.get_variable_scope().name, sub_name)
        if len(tf.get_collection(key, scope_name)) == 0:
            return tf.variable_scope(sub_name)
        i += 1



