import tensorflow as tf
import sys

def Print(input_, data = None, message=None, first_n=None, summarize=tf.int32.max, name=None):
    if data is None: data = [input_]
    return tf.Print(input_, data, message, first_n, summarize, name)
