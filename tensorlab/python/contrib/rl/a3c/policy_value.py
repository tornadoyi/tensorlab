import tensorflow as tf
from tensorlab.python.common import *
from tensorlab.python.ops.base import array as array_ops


class Policy(object):
    def __init__(self, build_net_func=None):
        self._build_net_func = build_net_func

        self._pi = None
        self._log_pi = None
        self._entropy = None
        self._predict_actions = None
        self._predict_action_probs = None



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


    def build(self, *args, **kwargs): self._build_network(*args, **kwargs)

    def _build_network(self, *args, **kwargs):
        if self._build_net_func is None: raise NotImplementedError("build_network function must be implementated")
        return self._build_net_func(*args, **kwargs)




class BasicPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super(BasicPolicy, self).__init__(*args, **kwargs)


    def build(self, input_states, input_actions, *args, **kwargs):
        self._pi = self._build_network(input_states, input_actions, *args, **kwargs)
        self._log_pi = tf.log(tf.maximum(self._pi, epsilon))
        self._entropy = -tf.reduce_sum(self._pi * self._log_pi, reduction_indices=1)
        self._predict_actions = tf.ones_like(self._pi)
        self._predict_action_probs = self._pi



class DistributionPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super(DistributionPolicy, self).__init__(*args, **kwargs)
        self._distribution = None


    def build(self, input_states, input_actions, *args, **kwargs):
        self._pi = self._distribution.prob(input_actions)
        self._log_pi = self._distribution.log_prob(input_actions)
        self._entropy = self._distribution.entropy()



class NormalDistributionPolicy(DistributionPolicy):
    def __init__(self,
                 validate_args=False,
                 allow_nan_stats=True,
                 *args, **kwargs
                 ):
        super(NormalDistributionPolicy, self).__init__(*args, **kwargs)
        self._validate_args = validate_args
        self._allow_nan_stats = allow_nan_stats


    def build(self, input_states, input_actions, *args, **kwargs):
        mu, sigma = self._build_network(input_states, input_actions, *args, **kwargs)
        self._distribution = tf.contrib.distributions.Normal(mu, sigma, self._validate_args, self._allow_nan_stats)

        super(NormalDistributionPolicy, self).build(input_states, input_actions, *args, **kwargs)
        self._predict_actions = mu
        self._predict_action_probs = self._distribution.prob(mu)



class MultiPolicy(Policy):
    def __init__(self, polices, num_actions, *args, **kwargs):
        super(MultiPolicy, self).__init__(*args, **kwargs)
        self._polices = polices
        self._num_actions = num_actions

        assert len(polices) > 0 and len(polices) == len(num_actions)


    def build(self, input_states, input_actions, *args, **kwargs):
        st = 0
        for i in xrange(len(self._polices)):
            p = self._polices[i]
            n = self._num_actions[i]
            ed = st + n
            p.build(input_states, input_actions[:,st:ed], *args, **kwargs)

            self._pi = p.pi if i == 0 else tf.concat([self._pi, p.pi], axis=array_ops.dim(p.pi, -1)-1)
            self._log_pi = p.log_pi if i == 0 else tf.concat([self._log_pi, p.log_pi], axis=array_ops.dim(p.log_pi, -1)-1)
            self._entropy = p.entropy if i == 0 else tf.concat([self._entropy, p.entropy], axis=array_ops.dim(p.entropy, -1)-1)

            self._predict_actions = p.predict_actions if i == 0 else \
                tf.concat([self._predict_actions, p.predict_actions], axis=array_ops.dim(p.predict_actions, -1)-1)

            self._predict_action_probs = p.predict_action_probs if i == 0 else \
                tf.concat([self._predict_action_probs, p.predict_action_probs], axis=array_ops.dim(p.predict_action_probs, -1)-1)

            st = ed



class Value(object):
    def __init__(self, build_net_func=None):
        self._build_net_func = build_net_func
        self._v = None

    @property
    def v(self): return self._v

    def build(self, *args, **kwargs): self._build_network(*args, **kwargs)

    def _build_network(self, *args, **kwargs):
        if self._build_net_func is None: raise NotImplementedError("build_network function must be implementated")
        return self._build_net_func(*args, **kwargs)




class BasicValue(Value):
    def __init__(self, *args, **kwargs): super(BasicValue, self).__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        self._v = self._build_network(*args, **kwargs)


