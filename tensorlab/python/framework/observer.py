

class Observer(object):

    ATTR_SEARCH_PREFIX = ["", "_", "__"]

    def __init__(self, *args):
        self._keys = args
        self._tensors = []
        self._values = []

        # create key index dict
        self._key_index = {}
        for i in xrange(len(self._keys)):
            k = self._keys[i]
            assert not self._key_index.has_key(k)
            self._key_index[k] = i



    def __getattr__(self, name):
        index = self._key_index.get(name, None)
        if index is None: raise Exception("{0} attribute is not in observer".format(name))
        return self._values[index]


    @property
    def tensors(self): return self._tensors


    def build(self, obj):
        self._tensors = []

        for key in self._keys:
            s_keys = [pfx + key for pfx in self.ATTR_SEARCH_PREFIX]

            t = None
            for k in s_keys:
                if not hasattr(obj, k): continue
                t = getattr(obj, k)
                break

            if t is None:
                raise Exception('{0} attribute is not in {1}'.format(key, type(obj).__name__))

            self._tensors.append(t)



    def update(self, values):
        assert len(values) == len(self._keys)
        self._values = values