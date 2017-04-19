import tensorflow as tf
import time
import os
from types import FunctionType, MethodType
import pickle


class Trainer(object):
    def __init__(self,
                 sess,
                 init_variables=True,
                 checkpoint=None,
                 max_save_second = None,
                 max_save_epoch = None,
                 max_epoch = None):
        self._sess = sess
        self._init_variables = init_variables
        self._checkpoint = checkpoint
        self._max_save_second = max_save_second
        self._max_save_epoch = max_save_epoch
        self._max_epoch = max_epoch
        self._saver = tf.train.Saver()

        # state information
        self._epoch = 0

        # temp parms
        self._training = False


    def __call__(self, fetches, feed_dict=None, epoch_call=None):

        # init variables
        if self._init_variables: self._sess.run(tf.global_variables_initializer())

        # restore checkpoint
        self.load_checkpoint()

        # temp vars
        pre_save_time = time.time()
        self._training = True

        # train
        while self._training:
            # epoch +1
            self._epoch += 1

            # end by epoch
            if self._max_epoch is not None and self._epoch > self._max_epoch: break

            # get feed dict
            feed_dict_data = feed_dict
            if isinstance(feed_dict, FunctionType) or isinstance(feed_dict, MethodType):
                feed_dict_data = feed_dict()

            # run
            result = self._sess.run(fetches, feed_dict_data)

            # time
            curtime = time.time()

            # save checkpoint
            if (self._checkpoint is not None and curtime - pre_save_time >= self._max_save_second) or \
                (self._max_save_epoch is not None and self._epoch >= self._max_save_epoch):
                pre_save_time = curtime
                self.save_checkpoint()

            # call back
            if epoch_call is not None: epoch_call(result)




    @property
    def epoch(self): return self._epoch

    @property
    def training(self): return self._training

    def stop(self): self._training = False



    def load_checkpoint(self):
        if self._checkpoint is None: return

        # check dir valid
        dirname = os.path.dirname(self._checkpoint)
        if not os.path.isdir(dirname): return

        # check checkpoint files
        filename = os.path.basename(self._checkpoint)
        filelist = ["checkpoint", filename + ".data", filename + ".index", filename + ".meta", filename + ".state"]
        match = 0
        for name in os.listdir(dirname):
            for fn in filelist:
                if name.find(fn) < 0: continue
                match += 1
                break

        if match < len(filelist): return
        try:
            self._saver.restore(self._sess, self._checkpoint)
            state_file_path = self._checkpoint + ".state"
            with open(state_file_path) as f:
                state = pickle.loads(f.read())
                self._epoch = state["epoch"]
        except Exception, e:
            print(e)


    def save_checkpoint(self):
        if self._checkpoint is None: return
        dirname = os.path.dirname(self._checkpoint)
        if not os.path.isdir(dirname): os.makedirs(dirname)

        try:
            # save checkpoint
            self._saver.save(self._sess, self._checkpoint)

            # save state
            state = {
                "epoch": self._epoch
            }
            state_file_path = self._checkpoint + ".state"
            with open(state_file_path, "w") as f: f.write(pickle.dumps(state))


        except Exception,e:
            print(e)