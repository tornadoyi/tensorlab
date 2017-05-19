import copy
from types import FunctionType, MethodType, TypeType
import numpy as np
import tensorflow as tf
from tensorlab.python.common import *
from policy_value import Policy, Value
from actor_critic_layer import ACLayer
from tensorlab.python.layers import Layer
from tensorlab.python.ops.base import variable_scope as var_scope


# Asynchronous Advantage Actor Critic (A3C),
# Referecce https://arxiv.org/pdf/1602.01783.pdf


class ActorCritic(Layer):
    def __init__(self,
                 state_shape, num_actions,
                 policy, value,
                 input_layer = None,
                 entropy_beta=0.01,
                 critic_shrink_learning_rate=1.0,
                 global_ac=None,
                 optimizer=None,
                 *args, **kwargs
                 ):
        # check
        assert state_shape is not None and num_actions is not None
        assert isinstance(policy, Policy) and isinstance(value, Value)

        # parameters
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._policy = policy
        self._value = value
        self._input_layer = input_layer
        self._entropy_beta = entropy_beta
        self._critic_shrink_learning_rate = critic_shrink_learning_rate
        self._optimizer = optimizer
        self._global_ac = global_ac

        # state
        self._rnn_init_states = self._rnn_next_states = None

        # call base
        super(ActorCritic, self).__init__(once_build=True, *args, **kwargs)

        # build
        self.build()

    @property
    def input_state(self): return self._input_state

    @property
    def input_action(self): return self._input_action

    @property
    def input_reward(self): return self._input_R

    @property
    def train_vars(self): return self._train_vars

    @property
    def op_push(self): return self._op_push

    @property
    def op_pull(self): return self._op_pull

    @property
    def gradients(self): return self._grads

    @property
    def rnn_init_states(self): return self._rnn_init_states

    @property
    def rnn_next_states(self): return self._rnn_next_states

    @property
    def total_loss(self): return self._total_loss

    @property
    def policy_loss(self): return self._policy_loss

    @property
    def value_loss(self): return self._value_loss

    @property
    def predict_actions(self): return self._predict_actions

    @property
    def predict_action_probs(self): return self._predict_action_probs

    @property
    def td(self): return self._td

    @property
    def v(self): return self._value.v


    def _build(self):
        self._input_state = tf.placeholder(tf.float32, shape=[None] + list(self._state_shape), name="states")
        self._input_action = tf.placeholder(tf.float32, shape=[None, self._num_actions], name="actions")
        self._input_R = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")

        # build policy and value
        self._build_policy_value(self._input_state, self._input_action)
        policies = self._policy if isinstance(self._policy, (list, tuple)) else [self._policy]

        # collect predict_actions and predict_action_probs
        self._predict_actions = [p.predict_action for p in policies]
        self._predict_action_probs = [p.predict_action_prob for p in policies]


        # temporary difference (R-V) (input for policy)
        self._td = self._input_R - self._value.v

        # calculate policy loss
        self._policy_loss = 0
        for i in xrange(len(policies)):
            policy = policies[i]
            log_pi, entropy, action = policy.log_pi, policy.entropy, policy.action

            loss = -tf.reduce_mean(tf.reduce_sum(log_pi * action, axis=range(1, log_pi.shape.ndims), keep_dims=True) *
                                   self._td + entropy * self._entropy_beta)

            self._policy_loss += loss


        # value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        self._value_loss = tf.nn.l2_loss(self._td) / tf.to_float(tf.shape(self._td)[0]) * self._critic_shrink_learning_rate

        # total loss
        self._total_loss = self._policy_loss + self._value_loss

        # collect variables
        self._train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_default_graph()._name_stack)

        # calculate gradients
        self._grads = tf.gradients(self._total_loss, self._train_vars)

        # build pull, push op
        self._op_pull, self._op_push = self._build_pull_push()

        # collect rnn states
        self._collect_rnn_states()



    def _build_policy_value(self, input_states, input_actions):
        # build policy and value
        self._policy.build(input_states, input_actions)
        self._value.build(input_states, input_actions)



    def _build_pull_push(self):
        if self._global_ac is None: return None, None

        g_vars = self._global_ac.train_vars
        vars = self._train_vars

        op_pull = tf.group(*[tf.assign(vars[i], g_vars[i]) for i in xrange(len(g_vars))])
        op_push = None if self._optimizer is None else self._optimizer.apply_gradients(zip(self._grads, g_vars))

        return op_pull, op_push



    def _collect_rnn_states(self):

        rnn_states = []
        if self._input_layer: rnn_states += self._input_layer.rnn_states

        policies = self._policy if isinstance(self._policy, (list, tuple)) else [self._policy]
        for p in policies:
            rnn_states += p.rnn_states

        rnn_states += self._value.rnn_states

        self._rnn_init_states = [init for (init, next) in rnn_states]
        self._rnn_next_states = [next for (init, next) in rnn_states]











