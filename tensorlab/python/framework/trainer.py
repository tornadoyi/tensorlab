import time
import os
import pickle
import re
from types import FunctionType, MethodType
import tensorflow as tf
import numpy as np
import thread
from archive import Archive

class TrainerBase(object):
    def __init__(self,
                 sess,
                 init_variables=True,
                 checkpoint=None,
                 archive=None,
                 max_save_second = None,
                 max_save_epoch = None,
                 max_epoch = None,
                 save_with_epoch = False,
                 lock_for_save = False,
                 ):

        self._sess = sess
        self._init_variables = init_variables
        self._checkpoint = checkpoint
        self._max_save_second = max_save_second
        self._max_save_epoch = max_save_epoch
        self._max_epoch = max_epoch
        self._save_with_epoch = save_with_epoch
        self._lock_for_save = lock_for_save
        self._archive = archive if archive is not None else Archive()
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
    def session(self): return self._sess

    @property
    def epoch(self): return self._epoch

    @property
    def training(self): return self._training

    def stop(self): self._training = False


    def _train_loop(self, *args, **kwargs): raise NotImplementedError("_train function must be implementated")


    def _next_epoch(self):
        # lock
        if self._lock_for_save: self._lock.acquire()

        # epoch + 1
        cur_epoch = self._epoch
        self._epoch += 1

        # unlock
        if self._lock_for_save: self._lock.release()

        # save checkpoint
        curtime = time.time()
        if self._checkpoint is not None and \
                (self._max_save_second is not None and curtime - self._last_save_time >= self._max_save_second) or \
                (self._max_save_epoch is not None and cur_epoch - self._last_save_epoch >= self._max_save_epoch):
            self._last_save_time = curtime
            self._last_save_epoch = cur_epoch
            self._save_checkpoint()



    def _check_end(self):
        if not self._training: return True
        if self._max_epoch is not None and self._epoch > self._max_epoch: return True
        return False



    def _load_checkpoint(self):
        if self._checkpoint is None: return
        epoch = self._archive.restore(self._sess, self._checkpoint)
        if epoch is not None: self._epoch = epoch


    def _save_checkpoint(self):
        if self._checkpoint is None: return
        global_step = self._epoch if self._save_with_epoch else None
        self._archive.save(self._sess, self._checkpoint, global_step = global_step)



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
        
        
        
        