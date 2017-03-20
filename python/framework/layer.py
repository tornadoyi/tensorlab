
from node import Node

class Layer(Node):
    def __init__(self, op, *args, **kwargs):
        Node.__init__(self, op, 1, *args, **kwargs)

    def __str__(self):
        base = Node.__str__(self)
        output = "{0} {1}".format(base, self.op_name)
        return output


    @property
    def op_name(self): return self._core_func.func_name
