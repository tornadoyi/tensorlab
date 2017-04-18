import tensorflow as tf
import time


class Trainer(object):
    def __init__(self,
                 sess,
                 init_variables=True,
                 checkpoint=None,
                 max_save_second = 60,
                 max_epoch = None):
        self._sess = sess
        self._init_variables = init_variables
        self._checkpoint = checkpoint
        self._max_save_second = max_save_second
        self._max_epoch = max_epoch
        self._saver = tf.train.Saver()
        self._epoch = tf.Variable(0, dtype=tf.int32, trainable=False)


    def __call__(self, fetches, feed_dict=None, epoch_call=None):

        # init variables
        if self._init_variables: self._sess.run(tf.global_variables_initializer())

        # restore checkpoint
        if self._checkpoint is not None: self._saver.restore(self._sess, self._checkpoint)

        # temp vars
        pre_save_time = time.time()

        # train
        while True:
            # run
            result = self._sess.run(fetches, feed_dict)

            # time
            curtime = time.time()

            # save checkpoint
            if self._checkpoint is not None and curtime - pre_save_time > self._max_save_second:
                pre_save_time = curtime
                self._saver.save(self._sess, self._checkpoint)

            # call back
            if epoch_call is not None: epoch_call(result)


            # epoch +1
            self._epoch.assign_add(1)

            # end by epoch
            if self._max_epoch is not None and self._epoch.eval() >= self._max_epoch: break