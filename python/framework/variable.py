import tensorflow as tf
from node import Node


class Variable(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, tf.Variable, 0, *args, **kwargs)
