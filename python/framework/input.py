import tensorflow as tf
from node import Node


class Input(Node):
    def __init__(self, dtype, shape = None, **kwargs):
        Node.__init__(self, **kwargs)
        self._kwargs["dtype"] = dtype
        self._kwargs["shape"] = shape


    def __str__(self):
        base = Node.__str__(self)
        output = "{0} {1}".format(base, self.get_attr("shape"))
        return output


    def tensor(self, *args): return tf.placeholder(**self._kwargs)


