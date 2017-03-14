import os
from layer import Layer
from input import Input
from variable import Variable

class Model(object):
    def __init__(self):
        self._inputs = {}
        self._variables = {}
        self._nodes = {}
        self._outputs = []


    @property
    def inputs(self):
        ret = []
        for k, v in self._inputs.items():
            if type(v) != list:
                ret.append(v)
                continue
            for i in v: ret.append(i)
        return ret


    @property
    def variables(self):
        ret = []
        for k, v in self._variables.items():
            if type(v) != list:
                ret.append(v)
                continue
            for i in v: ret.append(i)
        return ret


    def input(self, name): return None if name == None else self._inputs.get(name)

    def variable(self, name): return None if name == None else self._variables.get(name)


    def add_input(self, *args, **kwargs):
        input = Input(*args, **kwargs)
        if input.name == None:
            list = self._inputs.get(None)
            if list == None:
                list = []
                self._inputs[None] = list
            list.append(input)
        else:
            if self._inputs.has_key(input.name):
                raise Exception("Repeated input name {0}".format(input.name))
            self._inputs[input.name] = input
        return input


    def add_variable(self, **kwargs):
        variable = Variable(**kwargs)
        if variable.name == None:
            list = self._variables.get(None)
            if list == None:
                list = []
                self._inputs[None] = list
            list.append(variable)
        else:
            if self._inputs.has_key(variable.name):
                raise Exception("Repeated variable name {0}".format(variable.name))
            self._variables[variable.name] = variable



    def run(self, sess, fetches = [], feed_dict = {} ):
        # feed dict
        input_dict = {}

        def add_input_node(node, v):
            assert node != None
            tensor = self._nodes.get(node)
            assert tensor != None
            input_dict[tensor] = v


        for k, v in feed_dict.items():
            if type(k) == str:
                node = self.input(k)
                if node == None: raise Exception("Can not find input {0}".format(k))
                add_input_node(node, v)

            else:
                add_input_node(k, v)

        # fetches
        node_list = []
        fetch_list = []

        def add_fetch_node(node):
            assert node != None
            node_list.append(node)
            tensor = self._nodes.get(node)
            assert tensor != None
            fetch_list.append(tensor)


        # variable
        for node in self.variables: add_fetch_node(node)

        # outputs
        for node in self._outputs: add_fetch_node(node)

        # fetches
        for k in fetches:
            if type(k) == str:
                node = self.input(k)
                if node == None: raise Exception("Can not find input {0}".format(k))
                add_fetch_node(node)
            else:
                add_fetch_node(node)

        # run
        result = sess.run(fetch_list, input_dict)

        # assemble result
        assert len(result) == len(node_list)
        result_dict = {}
        for i in xrange(len(result)): result_dict[node_list[i]] = result[i]
        return result_dict



    def summary(self):
        inputs = self.inputs

        def deep_search(node):
            if self._nodes.has_key(node): return

            node_input = None
            if isinstance(node, Input):
                assert len(node.lasts) == 0
                node_input = []
            else:
                # check inputs
                tmplist = []
                for input in node.lasts:
                    tensor = self._nodes.get(input)
                    if tensor == None: return
                    tmplist.append(tensor)
                node_input = tmplist

            try:
                tensor = node.tensor(*node_input)
            except Exception, e:
                raise Exception("create tensor for {0} error\n{1}".format(node, e))

            self._nodes[node] = tensor

            # check end
            if len(node.nexts) == 0:
                self._outputs.append(node)
                return

            # do nexts
            for next in node.nexts:
                deep_search(next)


        for input in inputs:
            deep_search(input)



    def save(self, filepath):
        if os.path.isfile(filepath):
            os.remove(filepath)


    def load(self, filepath):
        pass

