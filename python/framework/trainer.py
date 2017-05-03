import time
import os
import pickle
import re
from types import FunctionType, MethodType
import tensorflow as tf
import numpy as np
import thread

class TrainerBase(object):
    def __init__(self,
                 sess,
                 saver = None,
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
        self._saver = saver if saver is not None else tf.train.Saver()
        self._lock = thread.allocate_lock()

        # state information
        self._epoch = 0

        # temp parms
        self._training = False
        self._rex_steps = None
        if self._checkpoint is not None:
            ex = "^{0}-(\d*)".format(os.path.basename(self._checkpoint))
            self._rex_steps = re.compile(ex)


    def __call__(self, *args, **kwargs):
        assert self._training == False

        # init variables
        if self._init_variables: self._sess.run(tf.global_variables_initializer())

        # restore checkpoint
        self._load_checkpoint()

        # temp vars
        self._last_save_time = time.time()
        self._last_save_epoch = self._epoch
        self._training = True

        # enter next epoch
        self._epoch += 1

        # train
        self._train_loop(*args, **kwargs)




    @property
    def epoch(self): return self._epoch

    @property
    def training(self): return self._training

    def stop(self): self._training = False


    def _train_loop(self, *args, **kwargs): raise NotImplementedError("_train function must be implementated")


    def _next_epoch(self):
        # lock
        self._lock.acquire()

        # save checkpoint
        curtime = time.time()
        if self._checkpoint is not None and \
                (self._max_save_second is not None and curtime - self._last_save_time >= self._max_save_second) or \
                (self._max_save_epoch is not None and self._epoch - self._last_save_epoch >= self._max_save_epoch):
            self._last_save_time = curtime
            self._last_save_epoch = self._epoch
            self._save_checkpoint()

        self._epoch += 1

        # unlock
        self._lock.release()


    def _check_end(self):
        if not self._training: return True
        if self._max_epoch is not None and self._epoch > self._max_epoch: return True
        return False



    def _load_checkpoint(self):
        if self._checkpoint is None: return

        # check dir valid
        filename, epoch = self._lastest_checkpoint()
        if filename is None: return

        try:
            self._saver.restore(self._sess, filename)
            if epoch is not None: self._epoch = int(epoch)

        except Exception, e:
            print(e)


    def _save_checkpoint(self):
        if self._checkpoint is None: return
        dirname = os.path.dirname(self._checkpoint)
        if not os.path.isdir(dirname): os.makedirs(dirname)

        try:
            # save checkpoint
            global_step = self._epoch if self._save_with_epoch else None
            self._saver.save(self._sess, self._checkpoint, global_step=global_step)

        except Exception,e:
            print(e)


    def _lastest_checkpoint(self):

        def split_epoch(filename):
            filename = os.path.basename(filename)
            steps = re.findall(self._rex_steps, filename)
            if len(steps) == 0: return None
            epoch = steps[0]#int(steps[0])
            return epoch

        # check dir valid
        dirname = os.path.dirname(self._checkpoint)
        if not os.path.isdir(dirname): return None, None

        # check latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(dirname)
        if latest_checkpoint is not None: return latest_checkpoint, split_epoch(latest_checkpoint)


        # check max epoch and valid checkpoint
        filename = os.path.basename(self._checkpoint)

        max_epoch = None
        max_epoch_file = None
        for fname in os.listdir(dirname):
            epoch = split_epoch(fname)
            if epoch is None: continue
            if max_epoch is None or int(epoch) > int(max_epoch):
                max_epoch = epoch
                max_epoch_file = os.path.splitext(os.path.join(dirname, fname))[0]

        return max_epoch_file, max_epoch




class Trainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

    def _train_loop(self, fetches, feed_dict=None, epoch_callback=None):
        # train
        while not self._check_end():

            # get feed dict
            feed_dict_data = feed_dict
            if isinstance(feed_dict, FunctionType) or isinstance(feed_dict, MethodType):
                feed_dict_data = feed_dict()

            # run
            result = self._sess.run(fetches, feed_dict_data)

            # call back
            if epoch_callback is not None: epoch_callback(result)

            # next epoch
            self._next_epoch()
        
        
        
        