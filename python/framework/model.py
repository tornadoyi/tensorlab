import os
from tensor import Tensor
from types import TypeType


class Model(object):
    def __init__(self):
        self._tensors = []

    def __iter__(self):
        for t in self._tensors: yield t

    def __len__(self): return len(self._tensors)

    def __getitem__(self, idx): return self._tensors[idx]


    @property
    def out(self):
        if len(self._tensors) == 0: return None
        return self._tensors[len(self._tensors)-1].tensor


    @property
    def tensors(self):
        return [t.tensor for t in self._tensors]


    def add(self, tensor_func, *args, **kwargs):
        '''
        if isinstance(tensor_func, Tensor):
            tensor = tensor_func
        elif isinstance(tensor_func, TypeType):
            tensor = tensor_func(*args, **kwargs)
        else:
            tensor = Tensor(tensor_func, *args, **kwargs)
        '''
        tensor = Tensor(tensor_func, *args, **kwargs)
        self._tensors.append(tensor)
        return tensor


    def tensor(self, index):
        if index >= len(self._tensors): raise Exception("index {0} is out of range".format(index))
        return self._tensors[index].tensor


    def run(self, sess, fetches=[], feed_dict={}):
        result_dict = {self.out: None}
        for t in fetches: result_dict[t] = None

        fetches = [k for k, v in result_dict.items()]
        result = sess.run(fetches, feed_dict)
        for i in xrange(len(result)):
            k, v = fetches[i], result[i]
            result_dict[k] = v

        return result_dict






