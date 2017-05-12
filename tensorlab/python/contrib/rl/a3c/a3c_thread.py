import copy
from types import FunctionType, MethodType, TypeType
import tensorflow as tf
import numpy as np
from tensorlab.python.common import *
from tensorlab.python.contrib.rl import Environment
from actor_critic import *


# Asynchronous Advantage Actor Critic (A3C),
# Referecce https://arxiv.org/pdf/1602.01783.pdf

class A3CThread(object):
    def __init__(self,
                 state_shape,
                 num_actions,
                 ac_kernel,
                 choose_action_func,
                 train_per_nsteps=1,
                 exploration_rate = 0.0,
                 reward_gamma = 0.9,
                 no_reward_at_terminal=True,
                 reward_clip = None,
                 ):

        self._ac_kernel = ac_kernel
        self._choose_action_func = choose_action_func
        self._running = False
        self._train_per_nsteps = train_per_nsteps
        self._exploration_rate = tf.convert_to_tensor(exploration_rate)
        self._reward_gamma = reward_gamma
        self._no_reward_at_terminal = no_reward_at_terminal
        self._reward_clip = reward_clip


        # build kernel
        self._input_state = tf.placeholder(tf.float32, shape=[None] + list(state_shape))
        self._input_action = tf.placeholder(tf.float32, shape=[None, num_actions])
        self._input_reward = tf.placeholder(tf.float32, shape=[None, 1])
        self._ac_kernel.build(self._input_state, self._input_action, self._input_reward)



    @property
    def running(self): return self._running

    @property
    def ac_kernel(self): return self._ac_kernel

    @property
    def exploration_rate(self): return self._exploration_rate

    @exploration_rate.setter
    def exploration_rate(self, v):  self._exploration_rate = v

    def stop(self): self._running = False


    def train(self, sess, env, step_callback=None, train_callback=None):
        # parms
        kernel = self._ac_kernel
        input_state = self._input_state
        input_action = self._input_action
        input_reward = self._input_reward

        # init
        self._running = True
        steps = 1
        s = env.reset()
        s_, r, a, t = None, None, None, False

        # run
        while self._running:

            # Reset gradients: d_theta <- 0 and d_theta_v <- 0.
            # Synchronize thread-specific parameters theta' = theta and theta_v' = theta_v
            sess.run([kernel.op_reset, kernel.op_pull])

            # t_start = t
            # Get state s_t
            step_start = steps

            # buffers
            states = []
            actions = []
            rewards = []
            terminals = []

            # check ternimal
            if t:
                s = env.reset()
                t = False

            while not t and steps - step_start < self._train_per_nsteps:

                # choose action
                a = self._choose_action(sess, env, s)
                s_, r, t = env.step(a)
                if self._reward_clip: r = np.clip(r, *self._reward_clip)

                # collect
                states.append(s)
                actions.append(a)
                rewards.append(r)
                terminals.append(t)

                # set state to next state
                s = s_

                # step once, and callback
                steps += 1
                if step_callback: step_callback(self)

                # end condition
                if not self._running: break


            # calculate R
            R = 0 if t and self._no_reward_at_terminal else self._calculate_R(sess, s_)
            Rs = np.zeros(len(states))
            for i in xrange(len(states)-1, -1, -1):
                R = rewards[i] + self._reward_gamma * R
                Rs[i] = R


            # train
            buf_s, buf_a, buf_R = np.vstack(states), np.vstack(actions), np.vstack(Rs)
            feed_dict = {
                input_state: buf_s,
                input_action: buf_a,
                input_reward: buf_R,
            }

            # collect train ops
            train_ops = [kernel.op_train]
            if kernel.observer is not None: train_ops += kernel.observer.tensors

            # train
            resulut = sess.run(train_ops, feed_dict=feed_dict)

            # update observer
            if kernel.observer is not None: kernel.observer.update(resulut[1:])

            # Perform asynchronous update of theta using d_theta and of theta_v using d_theta_v
            sess.run(kernel.op_push)

            # callback
            if train_callback: train_callback(self)



    def predict(self, sess, env, s):
        kernel = self._ac_kernel
        input_state = self._input_state
        actions, action_probs = sess.run([kernel.policy.predict_actions, kernel.policy.predict_action_probs],
                                         feed_dict={input_state: s})

        return self._choose_action_func(env, s, actions, action_probs)



    def _choose_action(self, sess, env, s):
        kernel = self._ac_kernel
        input_state = self._input_state

        if np.random.random() < sess.run(self._exploration_rate):
            return env.random_action(s)

        else:
            if self._input_state.shape.ndims != np.ndim(s): s = np.expand_dims(s, 0)
            actions, action_probs = sess.run([kernel.policy.predict_actions, kernel.policy.predict_action_probs],
                                      feed_dict = {input_state: s})

            actions = np.squeeze(actions)
            action_probs = np.squeeze(action_probs)

            return self._choose_action_func(env, s, actions, action_probs)



    def _calculate_R(self, sess, s):
        kernel = self._ac_kernel
        input_state = self._input_state
        batch_s = np.expand_dims(s, axis=0)

        R = sess.run(kernel.value.v, feed_dict = {input_state: batch_s})
        return np.squeeze(R)
