import os
import re
import tensorflow as tf


class Archive(object):
    def __init__(self, *args, **kwargs):
        self._saver = tf.train.Saver(*args, **kwargs)



    def restore(self, sess, save_path, epoch=None):

        # check dir valid
        if epoch is None:
            filename, epoch = self._lastest_checkpoint(save_path)
        else:
            filename = save_path + str(epoch)

        if filename is None: return None

        self._saver.restore(sess, filename)
        return int(epoch) if epoch is not None else None



    def save(self, *args, **kwargs):
        # check save path
        save_path = args[1] if len(args) >= 2 else kwargs.get('save_path')
        assert save_path is not None
        dirname = os.path.dirname(save_path)
        if not os.path.isdir(dirname): os.makedirs(dirname)

        # save checkpoint
        self._saver.save(*args, **kwargs)



    def _lastest_checkpoint(self, checkpoint):
        # use re
        rex_steps = re.compile("^{0}-(\d*)".format(os.path.basename(checkpoint)))

        def split_epoch(filename):
            filename = os.path.basename(filename)
            steps = re.findall(rex_steps, filename)
            if len(steps) == 0: return None
            epoch = steps[0]#int(steps[0])
            return epoch



        # check dir valid
        dirname = os.path.dirname(checkpoint)
        if not os.path.isdir(dirname): return None, None

        # check latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(dirname)
        if latest_checkpoint is not None: return latest_checkpoint, split_epoch(latest_checkpoint)


        # check max epoch and valid checkpoint
        filename = os.path.basename(checkpoint)

        max_epoch = None
        max_epoch_file = None
        for fname in os.listdir(dirname):
            epoch = split_epoch(fname)
            if epoch is None: continue
            if max_epoch is None or int(epoch) > int(max_epoch):
                max_epoch = epoch
                max_epoch_file = os.path.splitext(os.path.join(dirname, fname))[0]

        return max_epoch_file, max_epoch