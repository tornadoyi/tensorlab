

class Node(object):
    def __init__(self, **kwargs):
        self._nexts = []
        self._lasts = []
        self._kwargs = kwargs

    def __str__(self):
        output = "({0})".format(type(self).__name__.split(".")[-1])
        if self.name != None: output += self.name
        return output


    def tensor(self, *args): raise Exception("Not implementation")

    @property
    def lasts(self): return self._lasts

    @property
    def nexts(self): return self._nexts


    @property
    def name(self): return self._kwargs.get('name')


    @property
    def head(self):
        length = len(self._lasts)
        if length == 0: return self
        if length == 1: return self._lasts[0].head
        raise Exception("Brach node can not search")

    @property
    def tail(self):
        length = len(self._nexts)
        if length == 0: return self
        if length == 1: return self._nexts[0].tail
        raise Exception("Brach node can not search")



    def get_attr(self, name): return self._kwargs.get(name)


    def add(self, node):
        self._nexts.append(node)
        node._lasts.append(self)
        return node


    def adds(self, nodes):
        pre_node = self
        for node in nodes:
            pre_node.add(node)
            pre_node = node
        return pre_node


