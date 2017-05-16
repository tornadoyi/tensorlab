import gym
from gym.spaces.discrete import Discrete
import numpy as np
from env import Environment

class GymEnvironment(Environment):
    def __init__(self, game):
        super(GymEnvironment, self).__init__()
        self._game = game
        self._env = gym.make(game)

        self._action_space = self._env.action_space
        self._is_discrete = isinstance(self._action_space, Discrete)

        self._state_shape = self._env.observation_space.shape
        self._num_actions = self._action_space.n if self._is_discrete else self._action_space.shape[0]
        self._action_range = (0, 1) if self._is_discrete else (self._action_space.low, self._action_space.high)

        # current game state
        self._state = None
        self._terminal = False
        self._total_reward = 0


    @property
    def state_shape(self): return self._state_shape

    @property
    def num_actions(self): return self._num_actions

    @property
    def action_range(self): return self._action_range

    @property
    def terminal(self): return self._terminal

    @property
    def total_reward(self): return self._total_reward


    def step(self, action):
        a = np.argmax(action) if self._is_discrete else action
        s, r, t, _ =  self._env.step(a)
        self._state = s
        self._total_reward += r
        self._terminal = t
        return s, r, t

    def reset(self):
        self._state = None
        self._terminal = False
        self._total_reward = 0
        return self._env.reset()

    def close(self): return self._env.close()

    def seed(self, seed=None): return self._env.seed(seed)

    def render(self, mode='human', close=False): return self._env.render(mode, close)

    def random_action(self, s):
        action = self._action_space.sample()
        if self._is_discrete:
            a = np.zeros(self.num_actions, np.float32)
            a[action] = 1.0
            return a
        return action