import os
import time
import multiprocessing
import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.python.contrib.rl.a3c import BasicPolicy, MultiPolicy, BasicValue, ActorCritic, ActorCriticRNN, A3CThread, A3CTrainer
from tensorlab.python.contrib.rl.env import GymEnvironment
import matplotlib.pyplot as plt


ENTROPY_BETA = 0.01

CRITIC_SHRINK_LEARNING_RATE = 1.0

TRAIN_PER_NSTEPS = 5

EXPLORATION_RATE = 0#0.5

REWARD_GAMMA = 0.9

NO_REWARD_AT_TERMINAL = True

NUM_WORK_THREADS = multiprocessing.cpu_count() * 2

LEARNING_RATE = 0.001#1e-6

RMS_DECAY = 0.99

MAX_TRAIN_EPOCH = 1e+10

CHECK_POINT_PATH = "checkpoints/{0}/a3c-{1}/archive.ckpt"  # {0} is net type

SAVE_PER_SECOND = 20

GRAPH_SAVE_PATH = 'logs'



class A3C(object):
    def __init__(self, game, net_type="ff"):
        # parms
        self._game = game
        self._net_type = net_type


        # tools
        self._graph = tf.Graph()
        self._sess = tf.InteractiveSession(graph=self._graph)
        self._optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=RMS_DECAY, epsilon=0.1)

        # shape and num actions
        env = GymEnvironment(self._game)
        self._state_shape = env.state_shape
        self._num_actions = env.num_actions

        # create master and archive
        self._master, self._archive = self._build_master_archive()
        self._trainer = None


    def train(self, log_per_second=None, save_graph_per_second=None, save_statistic_per_second=None):
        # create trainer
        if self._trainer is None: self._trainer = self._build_a3c_trainer()


        # times
        self._log_per_second = log_per_second
        self._save_graph_per_second = save_graph_per_second
        self._save_statistic_per_second = save_statistic_per_second

        self._pre_log_time = time.time()
        self._pre_save_graph_time = time.time()
        self._pre_save_statistic_time = time.time()
        self._graph_writer = tf.summary.FileWriter(GRAPH_SAVE_PATH, self._graph)
        self._rewards = [] # (epoch, reward)

        # start train
        self._trainer(epoch_callback = lambda *args, **kwargs: self._epoch_callback(*args, **kwargs),
                      train_callback=lambda *args, **kwargs: self._train_callback(*args, **kwargs))


    def _epoch_callback(self, index):
        pass


    def _train_callback(self, index):
        if index != 0: return
        #print("")

        # get parms
        sess = self._trainer.session
        thread = self._trainer.get_train_thread(index)
        env = self._trainer.get_env(index)
        observer = thread.ac_kernel.observer

        # current time
        curtime = time.time()

        # print log
        if env.terminal and self._log_per_second is not None and curtime - self._pre_log_time >= self._log_per_second:
            self._pre_log_time = curtime
            print("loss: {0}  reward: {1}".format(observer.total_loss, env.total_reward))

        # save graph
        if self._save_graph_per_second is not None and curtime - self._pre_save_graph_time >= self._save_graph_per_second:
            self._pre_save_graph_time = curtime
            if not os.path.isdir(GRAPH_SAVE_PATH): os.makedirs(GRAPH_SAVE_PATH)
            self._graph_writer.flush()

        # save rewards
        if env.terminal: self._rewards.append((self._trainer.epoch ,env.total_reward))

        # save statistic
        if self._save_statistic_per_second is not None and curtime - self._pre_save_statistic_time >= self._save_statistic_per_second:
            self._pre_save_statistic_time = curtime
            if not os.path.isdir(GRAPH_SAVE_PATH): os.makedirs(GRAPH_SAVE_PATH)
            plt.plot([r[0] for r in self._rewards], [r[1] for r in self._rewards])
            plt.xlabel('epoch')
            plt.ylabel('total reward')
            plt.savefig(os.path.join(GRAPH_SAVE_PATH, 'rewards_{0}.png'.format(self._trainer.epoch)))
            #plt.show()



    def _choose_action(self, env, s, actions, action_probs):

        index = np.random.choice(len(action_probs), p=action_probs)
        action = np.zeros_like(actions)
        action[index] = 1.0

        return action




    def _build_a3c_trainer(self):
        # create slaves and envs of a3c threads
        with self._graph.as_default():
            master = self._master
            slaves = []
            envs = []
            for i in xrange(NUM_WORK_THREADS):
                slave = self._build_a3c_thread(self._optimizer, master)
                slaves.append(slave)

                env = GymEnvironment(self._game)
                envs.append(env)

            # create trainer
            trainer = A3CTrainer(
                master,
                slaves,
                envs,
                self._sess,
                checkpoint=CHECK_POINT_PATH.format(self._game, self._net_type),
                max_save_second=SAVE_PER_SECOND,
                save_with_epoch = True,
                archive=self._archive,
                verbose = True
            )
        return trainer


    def _build_master_archive(self):
        with self._graph.as_default():
            thread = self._build_a3c_thread(self._optimizer)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
            archive = tl.framework.Archive(var_list = vars)
        return thread, archive


    def _build_a3c_thread(self, optimizer, global_thread=None):
        global_ac = None if global_thread is None else global_thread.ac_kernel
        return A3CThread(
            self._state_shape,
            self._num_actions,
            self._build_actor_critic(optimizer, global_ac),
            lambda *args, **kwargs: self._choose_action(*args, **kwargs),
            train_per_nsteps=TRAIN_PER_NSTEPS,
            exploration_rate = EXPLORATION_RATE,
            reward_gamma = REWARD_GAMMA,
            no_reward_at_terminal = NO_REWARD_AT_TERMINAL,
        )


    def _build_actor_critic(self, optimizer, global_ac=None):

        w_init = tf.random_normal_initializer(0., .1)

        common_parms = {"optimizer" : optimizer,
                        "entropy_beta" : ENTROPY_BETA,
                        "critic_shrink_learning_rate" : CRITIC_SHRINK_LEARNING_RATE,
                        "global_ac" : global_ac,
                        "observer": tl.framework.Observer("total_loss", "policy_loss", "value_loss")}


        state_size = self._state_shape[0]
        action_size = self._num_actions


        if self._net_type == "rnn" or self._net_type == "lstm":

            class Policy(BasicPolicy):
                def _build_network(self):
                    layer = tf.layers.dense(self._states, 80, tf.nn.relu, kernel_initializer=w_init, name="policy_inputs")
                    return tf.layers.dense(layer, action_size, tf.nn.softmax, kernel_initializer=w_init, name="policy_output")


            class Value(BasicValue):
                def _build_network(self):
                    layer = tf.layers.dense(self._states, 50, tf.nn.relu, kernel_initializer=w_init, name="value_inputs")
                    return tf.layers.dense(layer, 1, None, kernel_initializer=w_init, name="value_outputs")

            if self._net_type == "rnn":
                cell = tf.contrib.rnn.BasicRNNCell(32)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(32)

            return ActorCriticRNN(cell,
                                  policy=Policy(),
                                  value=Value(),
                                  **common_parms)



        elif self._net_type == "ff":

            class Policy(BasicPolicy):
                def _build_network(self):
                    layer = tf.layers.dense(self._states, 200, tf.nn.relu, kernel_initializer=w_init, name="policy_inputs")
                    return tf.layers.dense(layer, action_size, tf.nn.softmax, kernel_initializer=w_init, name="policy_output")


            class Value(BasicValue):
                def _build_network(self):
                    layer = tf.layers.dense(self._states, 100, tf.nn.relu, kernel_initializer=w_init, name="value_inputs")
                    return tf.layers.dense(layer, 1, None, kernel_initializer=w_init, name="value_outputs")


            return ActorCritic(policy=Policy(),
                               value=Value(),
                               **common_parms)


        else: raise Exception("invalid net type " + self._net_type)

