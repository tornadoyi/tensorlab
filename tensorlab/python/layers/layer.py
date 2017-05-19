import tensorflow as tf

class Layer(object):
    def __init__(self, once_build=False, name=None):

        # paramaters
        self._once_build = once_build
        self._name = name
        self._scope = name if name is not None else type(self).__name__

        # build
        self._built = False
        self._outputs = None

    def __build__(self, *args, **kwargs): raise NotImplementedError("__build__ is not implemented")

    @property
    def outputs(self): return self._outputs

    @property
    def built(self): return self._built


    def build(self, *args, **kwargs):
        if self._once_build and self._built: raise Exception("{0} only build once".format(type(self).__name__))
        with tf.name_scope(self._scope):
            self._build(*args, **kwargs)
            self._built = True

    def _build(self, *args, **kwargs): self._outputs = self.__build__(*args, **kwargs)





