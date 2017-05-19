import tensorflow as tf
import numpy as np
from tensorlab.python.framework.trainer import TrainerBase
from actor_critic import ActorCritic
from actor_critic_player import ActorCriticPlayer
import threading



class A3CTrainer(TrainerBase):
    def __init__(self,
                 sess,
                 ac,
                 envs,
                 setting,
                 *args, **kwargs):

        # call base __init__
        super(A3CTrainer, self).__init__(sess, *args, **kwargs)

        # parms
        self._acs = ac if isinstance(ac, (list, tuple)) else [ac]
        self._settings = setting if isinstance(setting, (list, tuple)) else [setting] * len(self._acs)
        self._envs = envs
        self._threads = [A3CTrainThread(self._acs[i],
                                              self._settings[i],
                                              sess,
                                              self._envs[i]) for i in xrange(len(self._acs))]

        # check
        assert len(self._acs) == len(self._settings)
        assert len(self._acs) == len(self._envs)



    def get_ac(self, index): return self._acs[index]

    def get_env(self, index): return self._envs[index]

    def get_ac_thread(self, index): return self._threads[index]


    def _train_loop(self, epoch_callback=None, train_callback=None, wait_train_finish=True):

        def _on_once_step(i, t, env):
            # callback
            if epoch_callback: epoch_callback(i)

            # enter next epoch
            self._next_epoch()

            # check end
            if self._check_end():
                t.stop()
                return


        def _on_once_train(i, t, env):
            # callback
            if train_callback: train_callback(i)


        def _thread_wrapper(i):
            thread = self._threads[i]
            env = self._envs[i]
            thread.train(lambda *args: _on_once_step(i, thread, env),
                         lambda *args: _on_once_train(i, thread, env))



        worker_threads = []
        for i in xrange(len(self._threads)):
            t = threading.Thread(target=_thread_wrapper, args=(i, ))
            t.start()
            worker_threads.append(t)

        if wait_train_finish:
            coord = tf.train.Coordinator()
            coord.join(worker_threads)

        return worker_threads




# Asynchronous Advantage Actor Critic (A3C),
# Referecce https://arxiv.org/pdf/1602.01783.pdf

class A3CTrainThread(object):
    def __init__(self,
                 ac,
                 setting,
                 sess,
                 env
                 ):

        # parameters
        self._ac = ac
        self._setting = setting
        self._sess = sess
        self._env = env
        self._ac_player = ActorCriticPlayer(ac, sess, setting.choose_action)

        # runtime
        self._running = False


    @property
    def running(self): return self._running

    def stop(self): self._running = False


    def train(self, step_callback=None, train_callback=None):
        assert self.running is False
        # parms
        env = self._env
        player = self._ac_player
        setting = self._setting

        # init
        self._running = True
        s, s_, r, a, steps = [None] * 5
        t = True

        # run
        while self._running:
            # check ternimal
            if t:
                steps = 1
                s = env.reset()
                player.reset(steps)
                t = False


            # Reset gradients: d_theta <- 0 and d_theta_v <- 0.
            # Synchronize thread-specific parameters theta' = theta and theta_v' = theta_v
            player.pull()

            # t_start = t
            # Get state s_t
            step_start = steps

            # buffers
            states = []
            actions = []
            rewards = []
            terminals = []


            while not t and steps - step_start < setting.train_per_nsteps:

                # choose action
                if np.random.random() < setting.exploration_rate:
                    a = env.random_action(s)
                else:
                    a = player.choose_action(s, steps, env)


                # do action and get next state, reward and terminal
                s_, r, t = env.step(a)

                # clip reward
                if setting.reward_clip: r = np.clip(r, *setting.reward_clip)

                # save
                states.append(s)
                actions.append(a)
                rewards.append(r)
                terminals.append(t)

                # step finish, callback
                if step_callback: step_callback(self)

                # next step, set state to next state
                steps += 1
                s = s_

                # end condition
                if not self._running: break


            # calculate R
            R = 0 if t and setting.no_reward_at_terminal else player.predict_reward(s_, steps)
            Rs = np.zeros(len(states))
            for i in xrange(len(states)-1, -1, -1):
                R = rewards[i] + setting.reward_gamma * R
                Rs[i] = R


            # train
            buf_s, buf_a, buf_R = np.vstack(states), np.vstack(actions), np.vstack(Rs)
            player.train_one_step(buf_s, buf_a, buf_R, step_start)


            # Perform asynchronous update of theta using d_theta and of theta_v using d_theta_v
            player.push()

            # callback
            if train_callback: train_callback(self)






class A3CTrainSetting(object):
    def __init__(self,
                 choose_action,
                 train_per_nsteps=1,
                 exploration_rate=0.0,
                 reward_gamma=0.9,
                 no_reward_at_terminal=True,
                 reward_clip=None,
                 ):

        self._choose_action = choose_action
        self._train_per_nsteps = train_per_nsteps
        self._exploration_rate = exploration_rate
        self._reward_gamma = reward_gamma
        self._no_reward_at_terminal = no_reward_at_terminal
        self._reward_clip = reward_clip


    @property
    def choose_action(self): return self._choose_action

    @property
    def train_per_nsteps(self): return self._train_per_nsteps

    @property
    def exploration_rate(self): return self._exploration_rate

    @property
    def reward_gamma(self): return self._reward_gamma

    @property
    def no_reward_at_terminal(self): return self._no_reward_at_terminal

    @property
    def reward_clip(self): return self._reward_clip


    @exploration_rate.setter
    def exploration_rate(self, v): self._exploration_rate = v




