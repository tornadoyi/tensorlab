import inspect
import tensorflow as tf
from types import FunctionType, MethodType, TypeType

def _get_tf_tensor(tensor):
    if isinstance(tensor, Tensor): return tensor.tensor
    elif isinstance(tensor, tf.Tensor): return tensor
    else: raise Exception("type of tensor is {0},must be Tensor or tf.Tensor".format(type(tensor)))

class Tensor(object):
    def __init__(self, core_type, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._core_type = core_type
        self._core = core_type(*args, **kwargs)
        self._tf_tensor = _get_tf_tensor(self._core)
        self._arg_dict = {}
        self._transform = None

        if isinstance(core_type, FunctionType) or isinstance(core_type, MethodType):
            parse = inspect.getargspec(core_type)
            arg_names = parse.args
        elif isinstance(core_type, TypeType):
            parse = inspect.getargspec(core_type.__init__)
            arg_names = parse.args[1:] # drop self

        arg_defaults = parse.defaults or []

        for i in xrange(len(arg_names)):
            name = arg_names[i]
            if i < len(args):
                self._arg_dict[name] = args[i]

            elif kwargs.has_key(name):
                self._arg_dict[name] = kwargs[name]

            elif len(arg_names) - i <= len(arg_defaults):
                self._arg_dict[name] = arg_defaults[i-len(arg_names)]

            else:
                raise Exception("function {0} need parameter {1}".format(core_type.__name__, name))


    def __str__(self):
        output = "{0} {1} {2}".format(
            type(self).__name__.split(".")[-1],
            self.name,
            self.core_name
        )
        if self.name != None: output += self.name
        return output


    def __add__(self, other): return Tensor(lambda x,y: x + y, self.tensor, other)

    def __sub__(self, other): return Tensor(lambda x,y: x - y, self.tensor, other)

    def __mul__(self, other): return Tensor(lambda x,y: x * y, self.tensor, other)

    def __div__(self, other): return Tensor(lambda x,y: x / y, self.tensor, other)

    def __getitem__(self, item): return Tensor(lambda t,i: t[i], self.tensor, item)

    @property
    def core_type(self): return self._core_type

    @property
    def core(self): return self._core

    @property
    def tf_tensor(self): return self._tf_tensor

    @property
    def tensor(self): return self._tf_tensor

    @property
    def name(self): return self._tf_tensor.name

    @property
    def core_name(self): return self._core_type.__name__

    @property
    def shape(self): return self.tensor.get_shape()

    @property
    def transform(self): return self._transform

    def transform(self, f):
        self._transform = f
        if f is None:
            self._core = self._core_type(*self._args, **self._kwargs)
        else:
            self._core = f(self._tf_tensor)

        self._tf_tensor = _get_tf_tensor(self._core)
        return self

    def arg_attr(self, name): return self._arg_dict.get(name)