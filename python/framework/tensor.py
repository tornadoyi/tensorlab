import inspect

class Tensor(object):
    def __init__(self, tensor_func, *args, **kwargs):
        self._tensor_func = tensor_func
        self._tensor = tensor_func(*args, **kwargs)
        self._arg_dict = {}

        parse = inspect.getargspec(tensor_func)
        arg_names = parse.args
        arg_defaults = parse.defaults

        for i in xrange(len(arg_names)):
            name = arg_names[i]
            if i < len(args):
                self._arg_dict[name] = args[i]

            elif kwargs.has_key(name):
                self._arg_dict[name] = kwargs[name]

            elif len(arg_names) - i <= len(arg_defaults):
                self._arg_dict[name] = arg_defaults[i-len(arg_names)]

            else:
                raise Exception("function {0} need parameter {1}".format(tensor_func.__name__, name))


    def __str__(self):
        output = "{0} {1} {2}".format(
            type(self).__name__.split(".")[-1],
            self.name,
            self.func_name
        )
        if self.name != None: output += self.name
        return output

    @property
    def tensor(self): return self._tensor

    @property
    def name(self): return self._tensor.name


    @property
    def func_name(self): return self._tensor_func.__name__


    @property
    def shape(self): return self._tensor.get_shape()


    def arg_attr(self, name): return self._arg_dict.get(name)