import tensorflow as tf
from tensorlab.python.common import *
from tensorlab.python.ops.base import array as array_ops
from actor_critic_layer import ACLayer

class Policy(ACLayer):
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

        self._state = None
        self._action = None

        self._pi = None
        self._log_pi = None
        self._entropy = None
        self._predict_action = None
        self._predict_action_prob = None



    @property
    def state(self): return self._state

    @property
    def action(self): return self._action


    @property
    def pi(self): return self._pi

    @property
    def log_pi(self): return self._log_pi

    @property
    def entropy(self): return self._entropy

    @property
    def predict_action(self): return self._predict_action

    @property
    def predict_action_prob(self): return self._predict_action_prob

    def __build_inputs__(self, input_state, input_action): return (input_state, input_action)

    def _build(self, input_state, input_action):
        self._state, self._action = self.__build_inputs__(input_state, input_action)
        self._network = self.__build__(self._state, self._action)
        self._build_parameters()


    def _build_parameters(self): pass



class BasicPolicy(Policy):
    def __init__(self):
        super(BasicPolicy, self).__init__()


    def _build_parameters(self):
        self._pi = self._network
        self._log_pi = tf.log(tf.clip_by_value(self._pi, epsilon, 1.0))
        self._entropy = -tf.reduce_sum(self._pi * self._log_pi, axis=range(1, self._pi.shape.ndims))
        self._predict_action = tf.ones_like(self._pi)
        self._predict_action_prob = self._pi



class DistributionPolicy(Policy):
    def __init__(self):
        super(DistributionPolicy, self).__init__()
        self._distribution = None


    def _build_parameters(self):
        self._pi = self._distribution.prob(self._action)
        self._log_pi = self._distribution.log_prob(self._action)
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
        self._predict_action = self._mu
        self._predict_action_prob = self._distribution.prob(self._mu)





class Value(ACLayer):
    def __init__(self, *args, **kwargs):
        super(Value, self).__init__(*args, **kwargs)

        self._v = None

    @property
    def v(self): return self._v


    def _build(self, input_state, input_action):
        self._v = self.__build__(input_state, input_action)



class BasicValue(Value):
    def __init__(self): super(BasicValue, self).__init__()



