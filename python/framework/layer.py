
from node import Node

class Layer(Node):
    def __init__(self, op, **kwargs):
        Node.__init__(self, **kwargs)
        self._op = op

    def __str__(self):
        base = Node.__str__(self)
        output = "{0} {1}".format(base, self.op_name)
        return output


    @property
    def op_name(self): return self._op.func_name

    def tensor(self, *inputs): return self._op(*inputs, **self._kwargs)