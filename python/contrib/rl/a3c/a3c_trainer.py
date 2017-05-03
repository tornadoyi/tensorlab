import tensorflow as tf
import numpy as np
from python.framework.trainer import TrainerBase
import threading

class A3CTrainer(TrainerBase):
    def __init__(self,
                 master_thread,
                 train_threads,
                 sess,
                 var_list=None,
                 *args, **kwargs):

        # check
        assert master_thread is not None
        assert train_threads is not None and len(train_threads) > 0

        # call base __init__
        super(A3CTrainer, self).__init__(sess, *args, **kwargs)

        # parms
        self._master_thread = master_thread
        self._train_threads = train_threads
        self._coord = tf.train.Coordinator()





    def _train_loop(self, epoch_callback=None, train_callback=None):

        def _on_once_step(t):
            # callback
            if epoch_callback: epoch_callback(t)

            # enter next epoch
            self._next_epoch()

            # check end
            if self._check_end():
                t.stop()
                self._coord.request_stop()
                return


        def _on_once_train(t):
            # callback
            if train_callback: train_callback(t)


        def _thread_wrapper(thread):
            thread(self._sess, _on_once_step, _on_once_train)


        worker_threads = []
        for thread in self._train_threads:
            job = lambda: _thread_wrapper(thread)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)

        self._coord.join(worker_threads)