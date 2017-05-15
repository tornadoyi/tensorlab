import tensorflow as tf
from tensorlab.python.common import *
from tensorlab.python.ops.base import array as array_ops


class Policy(object):
    def __init__(self):

        self._states = None
        self._actions = None

        self._pi = None
        self._log_pi = None
        self._entropy = None
        self._predict_actions = None
        self._predict_action_probs = None

        self._log_pi_action = None

    @property
    def states(self): return self._states

    @property
    def actions(self): return self._actions


    @property
    def pi(self): return self._pi

    @property
    def log_pi(self): return self._log_pi

    @property
    def entropy(self): return self._entropy

    @property
    def predict_actions(self): return self._predict_actions

    @property
    def predict_action_probs(self): return self._predict_action_probs


    def build(self, input_states, input_actions):
        self._states, self._actions = self._build_inputs(input_states, input_actions)
        self._network = self._build_network()
        self._build_parameters()


    def _build_inputs(self, input_states, input_actions): return (input_states, input_actions)

    def _build_network(self): raise NotImplementedError("build_network function must be implementated")

    def _build_parameters(self): pass



class BasicPolicy(Policy):
    def __init__(self):
        super(BasicPolicy, self).__init__()


    def _build_parameters(self):
        self._pi = self._network
        self._log_pi = tf.log(tf.clip_by_value(self._pi, epsilon, 1.0))
        self._entropy = -tf.reduce_sum(self._pi * self._log_pi, axis=range(1, self._pi.shape.ndims))
        self._predict_actions = tf.ones_like(self._pi)
        self._predict_action_probs = self._pi



class DistributionPolicy(Policy):
    def __init__(self):
        super(DistributionPolicy, self).__init__()
        self._distribution = None


    def _build_parameters(self):
        self._pi = self._distribution.prob(self._actions)
        self._log_pi = self._distribution.log_prob(self._actions)
        self._entropy = self._distribution.entropy()



class NormalDistributionPolicy(DistributionPolicy):
    def __init__(self,
                 validate_args=False,
                 allow_nan_stats=True,
                 ):
        super(NormalDistributionPolicy, self).__init__()
        self._validate_args = validate_args
        self._allow_nan_stats = allow_nan_stats

        self._mu = None
        self._sigma = None


    def _build_parameters(self):
        self._mu, self._sigma = self._network
        self._distribution = tf.contrib.distributions.Normal(self._mu, self._sigma, self._validate_args, self._allow_nan_stats)
        super(NormalDistributionPolicy, self)._build_parameters()
        self._predict_actions = self._mu
        self._predict_action_probs = self._distribution.prob(self._mu)



class MultiPolicy(Policy):
    def __init__(self, polices):
        assert len(polices) > 0

        super(MultiPolicy, self).__init__()
        self._polices = polices



    def build(self, input_states, input_actions):
        self._states = []
        self._actions = []

        self._pi = []
        self._log_pi = []
        self._entropy = []
        self._predict_actions = []
        self._predict_action_probs = []

        for i in xrange(len(self._polices)):
            # pick state and actions
            p = self._polices[i]
            p.build(input_states, input_actions)

            self._states.append(p.states)
            self._actions.append(p.actions)
            self._pi.append(p.pi)
            self._log_pi.append(p.log_pi)
            self._entropy.append(p.entropy)
            self._predict_actions.append(p.predict_actions)
            self._predict_action_probs.append(p.predict_action_probs)



class Value(object):
    def __init__(self):
        self._states = None
        self._actions = None

        self._v = None

    @property
    def v(self): return self._v

    def build(self, input_states, input_actions):
        self._states = input_states
        self._actions = input_actions
        self._network = self._build_network()
        self._build_parameters()

    def _build_network(self, *args, **kwargs): raise NotImplementedError("build_network function must be implementated")

    def _build_parameters(self): self._v = self._network



class BasicValue(Value):
    def __init__(self): super(BasicValue, self).__init__()



