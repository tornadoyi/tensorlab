import tensorflow as tf

class _BaseLayer(object):
    def __init__(self, name=None):

        # paramaters
        self._name = name

        # build
        self._built = False
        self._outputs = None

    def __build__(self, *args, **kwargs): raise NotImplementedError("__build__ is not implemented")

    @property
    def outputs(self): return self._outputs

    @property
    def built(self): return self._built


    def build(self, *args, **kwargs):
        if self._built: raise Exception("{0} only build once".format(type(self).__name__))

        scope = self._variable_scope()
        if scope is None:
            self._build(*args, **kwargs)
        else:
            with scope:
                self._build(*args, **kwargs)
        self._built = True


    def _variable_scope(self): return None

    def _build(self, *args, **kwargs): self._outputs = self.__build__(*args, **kwargs)




class IndependentLayer(_BaseLayer):

    def _variable_scope(self):
        scope_name = self._name if self._name is not None else type(self).__name__
        return tf.variable_scope(None, default_name=scope_name, reuse=False)



class SharedLayer(_BaseLayer):

    def _variable_scope(self):
        scope_name = self._name if self._name is not None else type(self).__name__
        return tf.variable_scope(scope_name, reuse=self._built)



