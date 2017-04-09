import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import is_tensor

# stop
class _LoopState(list):
    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) == list:
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)

    @property
    def stop(self): return self[0]

    @stop.setter
    def stop(self, v): self[0] = v

    @property
    def should_stop(self): return self.stop

    def next_step(self): raise Exception("Need to implement")


# stop, cur_steop, start_step, end_step
class _ForState(_LoopState):
    def __init__(self, *args):
        _LoopState.__init__(self, *args)

    @property
    def step(self): return self[1]

    @property
    def start_step(self): return self[2]

    @property
    def end_step(self): return self[3]

    @property
    def should_stop(self): return tf.logical_or(tf.greater_equal(self.step, self.end_step), super(_ForState, self).stop)

    def next_step(self): self[1] += 1


# stop, cond
class _WhileState(_LoopState):
    def __init__(self, *args):
        _LoopState.__init__(self, *args)
        if self[1] is None: self[1] = tf.constant(True, tf.bool)

    @property
    def condition(self): return self[1]

    @property
    def should_stop(self): return tf.logical_or(tf.logical_not(self.condition), super(_WhileState, self).stop)

    def next_step(self): pass



def _loop_base(state, state_shape,
             body, loop_vars, shape_invariants=None,
             parallel_iterations=10, back_prop=True, swap_memory=False, name=None,
             auto_var_shape = False):

    # auto auto vars and shapes
    def recursive_var_shape(var):
        if isinstance(var, collections.Sequence):
            seq = []
            for v in var: seq.append(recursive_var_shape(v))
            return type(var)(seq)

        elif isinstance(var, tf.Tensor):
            dims = var.shape.ndims
            return tf.TensorShape([None]*dims)

        raise Exception("Invalid var type {0}".format(type(var)))

    if auto_var_shape:
        assert shape_invariants is None
        shape_invariants = []

        for var in loop_vars:
            shape = recursive_var_shape(var)
            shape_invariants.append(shape)


    # fill shape_invariants
    if shape_invariants is not None:
        assert len(loop_vars) == len(shape_invariants)
        shape_invariants = [state_shape] + shape_invariants


    # fill loop vars
    logic_var_count = len(loop_vars)
    loop_vars = [state] + loop_vars

    def _body(s, *loop_vars):
        loop_vars = body(s, *loop_vars)
        s.next_step()
        if logic_var_count > 1:
            return [s] + list(loop_vars)
        else:
            return [s, loop_vars]


    ret_value = tf.while_loop(lambda s, *_: tf.logical_not(s.should_stop),
                                _body, loop_vars, shape_invariants,
                                parallel_iterations, back_prop, swap_memory, name)
    values = ret_value[1:]
    return values[0] if len(values) == 1 else values



def for_loop(body, st, ed, delta=1, *args, **kwargs):
    # convert to tensor
    i = tf.constant(st, tf.int32) if not is_tensor(st) else st
    ed = tf.convert_to_tensor(ed) if not is_tensor(ed) else ed
    delta = tf.convert_to_tensor(delta) if not is_tensor(delta) else delta
    breaker = tf.constant(False, dtype=tf.bool)

    # state
    state = _ForState(breaker, i, i, ed)
    state_shape = _ForState([tf.TensorShape(None)] * 4)

    return _loop_base(state, state_shape, body, *args, **kwargs)


def while_loop(body, cond = None, *args, **kwargs):
    breaker = tf.constant(False, dtype=tf.bool)

    # state
    state = _WhileState(breaker, cond)
    state_shape = _WhileState([tf.TensorShape(None)] * 2)

    return _loop_base(state, state_shape, *args, **kwargs)



def logical_and(*conds):
    cur_cond = tf.constant(True, tf.bool)
    for cond in conds: cur_cond = tf.logical_and(cur_cond, cond)
    return cur_cond


def logical_or(*conds):
    cur_cond = tf.constant(False, tf.bool)
    for cond in conds: cur_cond = tf.logical_or(cur_cond, cond)
    return cur_cond