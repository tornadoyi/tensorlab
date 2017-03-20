import tensorflow as tf
from node import Node


class Input(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, tf.placeholder, 0, *args, **kwargs)


    def __str__(self):
        base = Node.__str__(self)
        output = "{0} {1}".format(base, self.shape)
        return output


    @property
    def dtype(self): return self.arg_attr("dtype")

    @property
    def shape(self): return self.arg_attr("shape")



