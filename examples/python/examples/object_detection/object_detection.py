from __future__ import division

import signal
import numpy as np
import tensorflow as tf

from tensorlab.framework.trainer import Trainer

from input import Input
from model import Model
from loss import mmod_loss

from ..support import dataset
from ..support.image import RandomCrop
from ..support.utils import *



def load_data(file):
    def create_rectangle(t, l, w, h):
        return [t, l, t+h-1 , l+w-1]

    images, labels = dataset.load_object_detection_xml(file, create_rectangle)
    return images, labels



def main(datapath):
    crop_size = (200, 200)
    mini_batch = 150
    detector_size = (40, 40)
    pyramid_scale = 6
    learning_rate = 1.0#1e-4

    # create session
    sess = tf.InteractiveSession()

    # load train datas
    images, labels = load_data(datapath)


    # create crop generator
    croper = RandomCrop(
        images, labels,
        set_chip_dims = crop_size,
        probability_use_label = 0.5,
        max_roatation_angle = 30,
        translate_amount = 0.1,
        random_scale_range = (1.3, 4),
        probability_flip = 0.5,
        min_rect_ratio = 0.01,
        min_part_rect_ratio = 0.4)


    # create input layer
    input_layer = Input(sess, pyramid_scale)

    # create model
    is_training = tf.Variable(False, dtype=tf.bool, trainable=False)
    model = Model(input_layer, is_training)

    # create loss
    loss = mmod_loss(model, detector_size)

    # create gradients and train step
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss.loss_tensor)
    train_step = optimizer.apply_gradients(gradients)

    # init all variables
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    # create trainer
    trainer = Trainer(sess, checkpoint="checkpoints/object-detection/od.ckpt", max_save_second=60)


    # ready train parameters
    # set train tag
    tf.assign(is_training, True).eval()
    print("set training state: {0}".format(is_training.eval()))

    # set fetches, feed dict and call back
    fetches = [train_step, loss.loss_tensor, gradients] + update_vars

    def gen_feed_dict():
        crop_images, crop_rects, rect_groups = croper(sess, mini_batch)
        return loss.gen_input_dict(crop_images, crop_rects, rect_groups)


    def train_on_step(result):
        v_loss = result[1]
        v_gradients = result[2]

        nan_inf_check_list = [v_loss, v_gradients]

        print("epoch: {0}".format(trainer.epoch))
        print("loss: ", v_loss)

        for i in xrange(len(gradients)):
            grad, var = gradients[i]
            v_gard, v_var = v_gradients[i]
            print("{0} grad:({1}, {2}) var({3}, {4})".format(var.name, np.min(v_gard), np.max(v_gard), np.min(v_var),
                                                             np.max(v_var)))

        for v in nan_inf_check_list:
            if not nan_inf_check(v): continue
            trainer.stop()
            break

        print("=" * 100)


    # start train
    trainer(fetches, gen_feed_dict, train_on_step)


    sess.close()



def nan_inf_check(v):
    if type(v) == list:
        for v_i in v:
            if nan_inf_check(v_i): return True
        return False

    else:
        exist = np.any(np.isnan(v)) or np.any(np.isinf(v))
        if exist:
            print(v)
            print("exist nan or inf")
            return True
        return False


if __name__ == "__main__":
    main()

    def signal_handler(signal, frame):
        global stop_requested
        print('You pressed Ctrl+C!')
        stop_requested = True
        exit()


    signal.signal(signal.SIGINT, signal_handler)




