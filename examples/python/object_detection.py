import math
import cv
import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab.framework import Model, layers
from util import *
from support import dataset
from support.image import RandomCrop
import time

def load_data(file):
    def create_rectangle(t, l, w, h):
        p = tl.Point2f(t, l)
        return tl.Rectanglef.create_with_tlwh(p, w, h)

    images, labels = dataset.load_object_detection_xml(file, create_rectangle)
    return images, labels



def create_model(input, is_training):
    def weight(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    '''
    def conv2d(x, W, b, strides, padding):
        return layers.conv2d(x, W, strides=strides, padding=padding).transform(lambda t: t + b)
    '''
    class conv2d(layers.conv2d):
        def __init__(self, x, W, b, strides, padding):
            layers.conv2d.__init__(self, x, W, strides=strides, padding=padding)
            self.transform(lambda t: t + b)

    def bn(inputs, is_traning):
        return tf.layers.batch_normalization(inputs, training=is_traning)


    model = Model()
    model.add(conv2d, input, weight([5, 5, 3, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
    model.add(bn, model.out, is_training)
    model.add(tf.nn.relu, model.out)

    model.add(conv2d, model.out, weight([5, 5, 32, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
    model.add(bn, model.out, is_training)
    model.add(tf.nn.relu, model.out)

    model.add(conv2d, model.out, weight([5, 5, 32, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
    model.add(bn, model.out, is_training)
    model.add(tf.nn.relu, model.out)

    model.add(conv2d, model.out, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
    model.add(bn, model.out, is_training)
    model.add(tf.nn.relu, model.out)

    model.add(conv2d, model.out, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
    model.add(bn, model.out, is_training)
    model.add(tf.nn.relu, model.out)

    model.add(conv2d, model.out, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
    model.add(bn, model.out, is_training)
    model.add(tf.nn.relu, model.out)

    model.add(conv2d, model.out, weight([6, 6, 32, 1]), bias([1]), [1, 1, 1, 1], padding="SAME")

    return model



def debug_images_rects(sess, images_tensor, rects_list):
    result = sess.run([images_tensor])
    images = result[0]
    for i in xrange(len(images)):
        img = images[i]
        rects = rects_list[i]
        for r in rects:
            cv2.rectangle(img, (int(r.left), int(r.top)), (int(r.right), int(r.bottom)), color=(0, 0, 255),
                          thickness=2)

        cv2.imshow("image", img)
        print("progress {0}/{1} rect:{2}".format(i + 1, len(images), len(rects)))
        press_key_stop()


class mmod_loss(object):
    def __init__(self,
                 model,
                 input_size,
                 detector_size,
                 loss_per_false_alarm = 1,
                 loss_per_missed_target = 1,
                 truth_match_iou_threshold = 0.5,):

        self._model = model
        self._input_size = input_size
        self._detector_size = detector_size
        self._loss_per_false_alarm = loss_per_false_alarm
        self._loss_per_missed_target = loss_per_missed_target
        self._truth_match_iou_threshold = truth_match_iou_threshold

        self._tensor_map_output_to_input = None


    def __call__(self, *args, **kwargs):
        pass


    def map_output_to_input(self, p):
        input_shape = p
        for layer in self._model:
            if layer.core_name == "conv2d":
                input_shape = layer.core.output_shape(input_shape)
                print(layer.core_name, input_shape)


    def gen_map_output_to_input_tensor(self):

        p = tf.placeholder(tf.int32, (2,))
        for layer in self._model:
            if layer.func_name != "conv2d": continue

            v_strides = layer.arg_attr("strides")
            v_padding = layer.arg_attr("padding")






def main():
    crop_size = (200, 200)
    crop_per_image = 30
    pyramid_scale = 6

    # load train datas
    images, labels = load_data("../data/training.xml")

    # create crop generator
    croper = RandomCrop(crop_size)

    # create model
    input = tf.placeholder(tf.float32, (None, None, None, None))
    is_training = tf.Variable(False, dtype=tf.bool)
    model = create_model(input, is_training)


    # create loss
    loss = mmod_loss(model, 40, 40)

    loss.map_output_to_input((150, 754, 200, 3))
    exit()


    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    while True:
        set_is_training = tf.assign(is_training, True)
        is_train = sess.run([set_is_training, is_training])[1]

        mini_batch_samples, mini_batch_labels = croper(images, labels, crop_per_image)
        mini_batch_samples = tl.image.pyramid(mini_batch_samples, pyramid_scale)
        #debug_images_rects(sess, mini_batch_samples, mini_batch_labels)


        mini_batch_samples = sess.run([mini_batch_samples])[0]

        model.run(sess, update_vars, feed_dict={input: mini_batch_samples})

        print("once finish")
        exit()


    sess.close()


if __name__ == "__main__":
    main()

