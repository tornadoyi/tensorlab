import inspect

class Node(object):
    def __init__(self, core_func, arg_index, *args, **kwargs):
        self._nexts = []
        self._lasts = []
        self._core_func = core_func
        self._arg_index = arg_index
        self._args = args
        self._kwargs = kwargs
        self._arg_dict = {}

        parse = inspect.getargspec(core_func)
        arg_names = parse.args
        arg_defaults = parse.defaults

        for i in xrange(len(arg_names)):
            name = arg_names[i]
            dft_index = i - len(arg_names)
            if i - arg_index < 0:
                self._arg_dict[name] = None
            elif i - arg_index < len(args):
                self._arg_dict[name] = args[i - arg_index]
            elif kwargs.has_key(name):
                self._arg_dict[name] = kwargs[name]
            elif -dft_index <= len(arg_defaults):
                self._arg_dict[name] = arg_defaults[dft_index]
            else:
                raise Exception("function {0} need parameter {1}".format(core_func.__name__, name))


    def __str__(self):
        output = "({0})".format(type(self).__name__.split(".")[-1])
        if self.name != None: output += self.name
        return output


    def tensor(self, *args, **kwargs):
        assert len(args) == self._arg_index
        all_args = args + self._args
        all_kwargs = dict(kwargs.items() + self._kwargs.items())
        return self._core_func(all_args, all_kwargs)


    @property
    def lasts(self): return self._lasts

    @property
    def nexts(self): return self._nexts


    @property
    def name(self): return self.arg_attr('name')


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



    def arg_attr(self, name): return self._arg_dict.get(name)


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


