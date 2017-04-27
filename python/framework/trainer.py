import time
import os
import pickle
import re
from types import FunctionType, MethodType
import tensorflow as tf

class Trainer(object):
    def __init__(self,
                 sess,
                 init_variables=True,
                 checkpoint=None,
                 max_save_second = None,
                 max_save_epoch = None,
                 max_epoch = None,
                 save_with_epoch = False):

        self._sess = sess
        self._init_variables = init_variables
        self._checkpoint = checkpoint
        self._max_save_second = max_save_second
        self._max_save_epoch = max_save_epoch
        self._max_epoch = max_epoch
        self._save_with_epoch = save_with_epoch
        self._saver = tf.train.Saver()

        # state information
        self._epoch = 0

        # temp parms
        self._training = False
        self._rex_steps = None
        if self._checkpoint is not None:
            ex = "^{0}-(\d*)\.".format(os.path.basename(self._checkpoint))
            self._rex_steps = re.compile(ex)


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
                file_max_epoch = self.find_max_epoch()
                history_max_epoch = state["epoch"]
                self._epoch = history_max_epoch if file_max_epoch < 0 else min(history_max_epoch, file_max_epoch)
        except Exception, e:
            print(e)


    def save_checkpoint(self):
        if self._checkpoint is None: return
        dirname = os.path.dirname(self._checkpoint)
        if not os.path.isdir(dirname): os.makedirs(dirname)

        try:
            # save checkpoint
            global_step = self._epoch if self._save_with_epoch else None
            self._saver.save(self._sess, self._checkpoint, global_step=global_step)

            # save state
            state = {
                "epoch": self._epoch
            }
            state_file_path = self._checkpoint + ".state"
            with open(state_file_path, "w") as f: f.write(pickle.dumps(state))


        except Exception,e:
            print(e)


    def find_max_epoch(self):
        max_epoch = -1
        dirname = os.path.dirname(self._checkpoint)
        if not os.path.isdir(dirname):return max_epoch
        if self._rex_steps is None: return max_epoch

        for fname in os.listdir(dirname):
            steps = re.findall(self._rex_steps, fname)
            if len(steps) == 0: continue
            epoch = int(steps[0])
            max_epoch = min(max_epoch, epoch)

        return max_epoch


