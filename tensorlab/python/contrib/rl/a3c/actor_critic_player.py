import numpy as np
import tensorflow as tf


class ActorCriticPlayer(object):
    def __init__(self, ac, sess, choose_action):

        # parameters
        self._sess = sess
        self._ac = ac
        self._choose_action = choose_action


        # runtime
        self._game_start_step = None
        self._step_rnn_states = []
        self._acc_grads = None


    def reset(self, step):
        assert type(step) is int
        self._game_start_step = step
        self._step_rnn_states = [self._sess.run(self._ac.rnn_init_states)]


    def predict_reward(self, s, step):
        ac = self._ac
        sess = self._sess

        # check
        rnn_state = self._find_rnn_state(step)

        # batch s
        assert ac.input_state.shape.ndims - 1 == np.ndim(s)
        batch_s = np.expand_dims(s, 0)

        # predict
        R, next_rnn_state = sess.run([ac.v, ac.rnn_next_states],
                     feed_dict=dict( [ac.input_state, batch_s] + zip(ac.rnn_init_states, rnn_state) ))

        # save rnn state
        self._save_rnn_state(next_rnn_state, step + len(s))

        return np.squeeze(R)



    def predict_action(self, s, step):
        ac = self._ac
        sess = self._sess

        # check
        rnn_state = self._find_rnn_state(step)

        # batch s
        assert ac.input_state.shape.ndims - 1 == np.ndim(s)
        batch_s = np.expand_dims(s, 0)

        # predict
        actions, action_probs, next_rnn_state = sess.run([
            ac.predict_actions, ac.predict_action_probs, ac.rnn_next_states],
            feed_dict=dict([(ac.input_state, batch_s)] + zip(ac.rnn_init_states, rnn_state) ))

        # save rnn state
        self._save_rnn_state(next_rnn_state, step + len(s))

        return np.squeeze(actions), np.squeeze(action_probs)


    def choose_action(self, s, step, *args, **kwargs):
        actions, action_probs = self.predict_action(s, step)
        return self._choose_action(actions, action_probs, *args, **kwargs)



    def train_one_step(self, s, a, r, step):
        ac = self._ac
        sess = self._sess

        # find rnn state
        rnn_state = self._find_rnn_state(step)

        # prepare fetches and feeds
        fetches = [ac.gradients]
        feeds = [(ac.input_state, s), (ac.input_aciton, a), (ac.input_reward, r)] + \
                zip(ac.rnn_init_states, rnn_state)

        # result
        results = sess.run(fetches, dict(feeds))
        grads = results[0]

        # accumulate grads
        if self._acc_grads is None:
            self._acc_grads = grads
        else:
            for i in xrange(len(self._acc_grads)):
                self._acc_grads[i] += grads[i]



    def push(self):
        assert self._acc_grads is not None
        self._sess.run(self._ac.op_push, {tuple(self._ac.gradients): self._acc_grads})
        self._acc_grads = None


    def pull(self): self._sess.run(self._ac.op_pull)



    def _find_rnn_state(self, step):
        index = step - self._game_start_step
        if index >= len(self._step_rnn_states):
            raise Exception("receive an invalid step {0}, current rnn states {1}s, from {2} ~ {3}",
                            step, len(self._step_rnn_states),
                            self._game_start_step, self._game_start_step + len(self._step_rnn_states) - 1)

        return self._step_rnn_states[index]


    def _save_rnn_state(self, state, step):
        index = step - self._game_start_step

        if index == len(self._step_rnn_states):
            self._step_rnn_states.append(state)

        elif index < len(self._step_rnn_states):
            pre_state = self._step_rnn_states[index]
            assert np.sum(state - pre_state) < 1e-6

        else:
            raise Exception("receive an invalid step {0}, current rnn states {1}s, from {2} ~ {3}",
                      step, len(self._step_rnn_states),
                      self._game_start_step, self._game_start_step + len(self._step_rnn_states) - 1)