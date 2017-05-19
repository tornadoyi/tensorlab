import tensorflow as tf
from tensorlab.python.layers import IndependentLayer


class ACLayer(IndependentLayer):
    def __init__(self, *args, **kwargs):
        super(ACLayer, self).__init__(*args, **kwargs)

        # rnn
        self._rnn_states = []


    @property
    def rnn_states(self): return self._rnn_states

    def _add_rnn_state(self, init_state, next_state): self._rnn_states.append((init_state, next_state))




class ACInput(ACLayer): pass


class ACInputRNN(ACInput):
    def __init__(self,
                 rnn_cell,
                 actor_update_rnn_grad=True,
                 critic_update_rnn_grad=True,
                 *args, **kwargs):


        # parameters
        self._rnn_cell = rnn_cell
        self._actor_update_rnn_grad = actor_update_rnn_grad
        self._critic_update_rnn_grad = critic_update_rnn_grad

        super(ACInputRNN, self).__init__(*args, **kwargs)




    def __build__(self, input_state):

        # initial rnn state
        init_rnn_state = self._rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

        # (time_major = False, so output shape is [batch_size=1, max_time, cell.output_size])
        input = tf.expand_dims(input_state, 0)
        output, next_rnn_state = tf.nn.dynamic_rnn(self._rnn_cell,
                                          input,
                                          initial_state=init_rnn_state,
                                          time_major=False,
                                          scope=None)

        # record rnn state
        self._add_rnn_state(init_rnn_state, next_rnn_state)


        # remove unused dim batch of lstm_outputs
        self._rnn_output = tf.squeeze(output, [0])


        # build policy and values
        policy_states = self._rnn_output if self._actor_update_rnn_grad else tf.stop_gradient(self._rnn_output)
        value_states = self._rnn_output if self._critic_update_rnn_grad else tf.stop_gradient(self._rnn_output)


        return policy_states, value_states