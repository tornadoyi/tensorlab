
class Environment(object):
    def __init__(self):
        pass

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__)

    def __repr__(self):
        return str(self)

    @property
    def state_shape(self): raise NotImplementedError("state_shape property must be implementated")

    @property
    def num_actions(self): raise NotImplementedError("num_actions property must be implementated")

    @property
    def action_range(self): raise NotImplementedError("action_range property must be implementated")


    def step(self, action): raise NotImplementedError("step method must be implementated")

    def reset(self): raise NotImplementedError("reset method must be implementated")

    def close(self): raise NotImplementedError("close method must be implementated")

    def seed(self, seed=None): pass

    def render(self, mode='human', close=False): pass

    def random_action(self, s): raise NotImplementedError("random_action method must be implementated")







