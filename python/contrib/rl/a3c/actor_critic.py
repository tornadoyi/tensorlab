import copy
from types import FunctionType, MethodType, TypeType
import tensorflow as tf
from python.common import *
from policy_value import Policy, Value

# Asynchronous Advantage Actor Critic (A3C),
# Referecce https://arxiv.org/pdf/1602.01783.pdf





class ActorCritic(object):

    def __init__(self,
                 policy,
                 value,
                 optimizer,
                 entropy_beta=0.01,
                 critic_shrink_learning_rate = 1.0,
                 global_ac = None,
                 ):

        # check
        self._local_scope = "{0}/{1}".format(tf.get_variable_scope(), type(self).__name__)
        if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._local_scope)) > 0:
            raise Exception("Repeated creation of {0} in scope {1}".format(type(self).__name__, tf.get_variable_scope()))

        assert isinstance(policy, Policy) or isinstance(policy, Value)


        # parameters
        self._policy = policy
        self._value = value
        self._entropy_beta = entropy_beta
        self._critic_shrink_learning_rate = critic_shrink_learning_rate
        self._optimizer = optimizer
        self._global_ac = global_ac

        # ops
        self._reset_with_ops = []
        self._train_with_ops = []
        self._op_reset = self._op_train = None

        # state
        self._has_build = False



    @property
    def policy(self): return self._policy

    @property
    def value(self): return self._value

    @property
    def train_vars(self): return self._train_vars

    @property
    def op_push(self): return self._op_push

    @property
    def op_pull(self): return self._op_pull

    @property
    def op_reset(self): return self._op_reset

    @property
    def op_train(self): return self._op_train



    def build(self, states, actions, R):
        """
        :param state: used on choose action, should be Tensor of shape: [batch, ...]
        :param action: used calculate policy loss, should be Tensor of shape: [batch, num_actions]
        :param R: nsteps reward should be Tensor of shape: [batch, 1]
        :return: 
        """

        # check
        assert self._has_build == False
        assert states is not None and states.shape.ndims == 2
        assert actions is not None and actions.shape.ndims == 2
        assert R is not None and R.shape.ndims == 2

        self._has_build = True
        self._states = states
        self._actions = actions
        self._R = R


        # build network
        with tf.variable_scope(self._local_scope):
            # build policy and value
            self._build_policy_value(self._states, self._actions)

            PI, log_pi, entropy  = self._policy.pi, self._policy.log_pi, self._policy.entropy
            V = self._value.v

            # temporary difference (R-V) (input for policy)
            self._TD = self._R - V

            # policy loss (output)  (Adding minus, because the original paper's objective
            # function is for gradient ascent, but we use gradient descent optimizer.)
            self._policy_loss = -tf.reduce_sum(tf.reduce_sum(log_pi * self._actions, axis=1) *
                                               self._TD + entropy * self._entropy_beta)

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            self._value_loss = tf.nn.l2_loss(self._TD) * self._critic_shrink_learning_rate

            # total loss
            self._total_loss = self._policy_loss + self._value_loss

            # collect variables
            self._train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._local_scope)

            # calculate gradients
            self._grads = tf.gradients(self._total_loss, self._train_vars)

            # accumulate gradients
            self._acc_grads, self._op_acc_grads, self._op_clean_grads = self._build_accumulate_gradients()

            # build pull, push op
            self._op_pull, self._op_push = self._build_pull_push()

            # add ops to reset and update list
            self._reset_with_ops.append(self._op_clean_grads)
            self._train_with_ops.append(self._op_acc_grads)
            self._op_reset = tf.group(*self._reset_with_ops)
            self._op_train = tf.group(*self._train_with_ops)



    def _build_accumulate_gradients(self):
        acc_grads = []
        op_acc_grads = []
        op_clean_grads = []
        for grad in self._grads:
            zero = tf.zeros_like(grad, grad.dtype)
            acc_grad = tf.Variable(zero, trainable=False, name="acc_{0}".format(grad.name))
            op_acc_grad = tf.assign_add(acc_grad, grad)
            op_clean_grad = tf.assign(acc_grad, zero)
            acc_grads.append(acc_grad)
            op_acc_grads.append(op_acc_grad)
            op_clean_grads.append(op_clean_grad)

        return acc_grads, tf.group(*op_acc_grads), tf.group(*op_clean_grads)



    def _build_pull_push(self):
        if self._global_ac is None: return [], [], [], []

        g_vars = self._global_ac.train_vars
        vars = self._train_vars
        acc_grads = self._acc_grads

        op_pull = tf.group(*[tf.assign(vars[i], g_vars[i]) for i in xrange(len(g_vars))])
        op_push = self._optimizer.apply_gradients(zip(acc_grads, g_vars))

        return op_pull, op_push



    def _build_policy_value(self, input_states, input_actions):
        # build policy and value
        self._policy.build(input_states, input_actions)
        self._value.build(input_states, input_actions)





class ActorCriticRNN(ActorCritic):
    def __init__(self,
                 rnn_cell,
                 actor_update_rnn_grad=True,
                 critic_update_rnn_grad=True,
                 *args, **kwargs):


        # parameters
        self._rnn_cell = rnn_cell
        self._actor_update_rnn_grad = actor_update_rnn_grad
        self._critic_update_rnn_grad = critic_update_rnn_grad

        # rnn properties
        self._rnn_state = self._initial_rnn_state = self._op_update_rnn_state = self._op_reset_rnn_state = None


        super(ActorCriticRNN, self).__init__(*args, **kwargs)


    @property
    def rnn_state(self): return self._rnn_state

    @property
    def initial_rnn_state(self): return self._initial_rnn_state

    @property
    def op_update_rnn_state(self): return self._op_update_rnn_state

    @property
    def op_reset_rnn_state(self): return self._op_reset_rnn_state


    def _build_policy_value(self, input_states, *args, **kwargs):

        self._initial_rnn_state = self._rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
        self._rnn_state = tf.Variable(self._initial_rnn_state, dtype=tf.float32, trainable=False)

        # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
        input = tf.expand_dims(input_states, 0)
        output, state = tf.nn.dynamic_rnn(self._rnn_cell,
                                          input,
                                          initial_state=self._rnn_state,
                                          time_major=False,
                                          scope=self._local_scope)

        # remove unused dim batch of lstm_outputs
        output = tf.reshape(output, [-1, tf.shape(output)[-1]])

        # cur_state op
        self._op_reset_state = tf.assign(self._rnn_state, self._initial_rnn_state)
        self._op_update_state = tf.assign(self._rnn_state, state)


        # build policy and values
        policy_states = output if self._actor_update_rnn_grad else tf.stop_gradient(output)
        value_states = output if self._critic_update_rnn_grad else tf.stop_gradient(output)

        self._policy.build(policy_states, *args, **kwargs)
        self._value.build(value_states, *args, **kwargs)