import tensorflow as tf
from node import Node


class Variable(Node):
    def __init__(self, initial_value, **kwargs):
        Node.__init__(self, **kwargs)
        self._kwargs["initial_value"] = initial_value



    def tensor(self, *args): return tf.Variable(**self._kwargs)