from __future__ import division

import math
import cv
import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.framework import layers
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



class Model(framework.Model):
    def __init__(self, input, is_training):
        framework.Model.__init__(self)
        self._gen_net(input, is_training)
        self._gen_output_shape_tensors()
        self._gen_padding_tensors()
        self._gen_map_input_to_output_tensor()
        self._gen_map_output_to_input_tensor()


    def output_shapes(self, sess, input_shape):
        return sess.run(self._output_shape_tensors, feed_dict={self._ph_input_shape: input_shape})

    def padding_values(self, sess, input_shape):
        return sess.run(self._padding_tensors, feed_dict={self._ph_input_shape: input_shape})

    def map_input_output(self, sess, input_shape, point):
        return sess.run(self._map_input_to_output_tensors,
                        feed_dict={self._ph_input_shape: input_shape,
                                   self._ph_input_point: point})

    def map_output_input(self, sess, input_shape, point):
        return sess.run(self._map_output_to_input_tensors,
                        feed_dict={self._ph_input_shape: input_shape,
                                   self._ph_input_point: point})


    def _gen_net(self, input, is_training):
        def weight(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        class conv2d(layers.conv2d):
            def __init__(self, x, W, b, strides, padding):
                layers.conv2d.__init__(self, x, W, strides=strides, padding=padding)
                self.transform(lambda t: t + b)

        def bn(inputs, is_traning):
            return tf.layers.batch_normalization(inputs, training=is_traning)


        self.add(conv2d, input, weight([5, 5, 3, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
        self.add(bn, self.out, is_training)
        self.add(tf.nn.relu, self.out)

        self.add(conv2d, self.out, weight([5, 5, 32, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
        self.add(bn, self.out, is_training)
        self.add(tf.nn.relu, self.out)

        self.add(conv2d, self.out, weight([5, 5, 32, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
        self.add(bn, self.out, is_training)
        self.add(tf.nn.relu, self.out)

        self.add(conv2d, self.out, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
        self.add(bn, self.out, is_training)
        self.add(tf.nn.relu, self.out)

        self.add(conv2d, self.out, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
        self.add(bn, self.out, is_training)
        self.add(tf.nn.relu, self.out)

        self.add(conv2d, self.out, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
        self.add(bn, self.out, is_training)
        self.add(tf.nn.relu, self.out)

        self.add(conv2d, self.out, weight([6, 6, 32, 1]), bias([1]), [1, 1, 1, 1], padding="SAME")


    def _gen_output_shape_tensors(self):
        self._ph_input_shape = tf.placeholder(dtype=tf.int32, shape=(4,))
        self._output_shape_tensors = []
        input_shape = self._ph_input_shape
        for layer in self:
            if layer.core_name == "conv2d":
                input_shape = layer.core.output_shape(input_shape)
                #print(layer.core_name, self._tensor_output_shape.eval())

            self._output_shape_tensors.append(input_shape)


    def _gen_padding_tensors(self):
        self._padding_tensors = []
        for i in xrange(len(self)):
            layer = self[i]
            if i == 0:
                input_shape = self._ph_input_shape
            else:
                input_shape = self._output_shape_tensors[i-1]
            input_shape = input_shape[1:3]

            if layer.core_name == "conv2d":
                padding = layer.core.padding_value(input_shape)
                self._padding_tensors.append(padding)
            else:
                self._padding_tensors.append(tf.constant((0,0), dtype=tf.int32))


    def _gen_map_input_to_output_tensor(self):
        self._ph_input_point = tf.placeholder(dtype=tf.int32, shape=[2,])
        self._map_input_to_output_tensors = []
        point = tf.reverse(self._ph_input_point, [0,])
        for i in xrange(len(self)):
            layer = self[i]
            if layer.core_name == "conv2d":
                padding = self._padding_tensors[i]
                filter = np.array(layer.core.filter[0:2], dtype=np.int)
                strides = np.array(layer.core.strides[1:3], dtype=np.int)
                point = (point + padding - tf.cast(filter/2, tf.int32)) / strides
                point = tf.cast(point, tf.int32)
                #print(filter, strides)

            self._map_input_to_output_tensors.append(tf.reverse(point, [0,]))


    def _gen_map_output_to_input_tensor(self):
        self._ph_input_point = tf.placeholder(dtype=tf.int32, shape=[2,])
        self._map_output_to_input_tensors = []
        point = tf.reverse(self._ph_input_point, [0,])
        for i in xrange(len(self)-1, -1, -1):
            layer = self[i]
            if layer.core_name == "conv2d":
                padding = self._padding_tensors[i]
                filter = np.array(layer.core.filter[0:2], dtype=np.int)
                strides = np.array(layer.core.strides[1:3], dtype=np.int)
                point = point * strides - padding + tf.cast(filter/2, tf.int32)

            self._map_output_to_input_tensors.append(tf.reverse(point, [0,]))



def debug_images_rects(sess, images_tensor, plans, rects_list):
    result = sess.run([images_tensor, plans])
    images, plans = result
    for i in xrange(len(images)):
        img = images[i]

        for p in plans[1:]:
            y, x, h, w = p
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color=(255, 0, 0), thickness=2)

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




    def __call__(self, *args, **kwargs):
        pass






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
    model = Model(input, is_training)

    # create loss
    loss = mmod_loss(model, 40, 40)

    # train
    # create session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    '''
    #for s in model.output_shapes(sess, (150, 754, 200, 3)): print s
    #print("="*100)
    #for s in model.padding_values(sess, (150, 754, 200, 3)): print s

    for i in xrange(100):
        for j in xrange(100):
            point = model.map_output_input(sess, (150, 754, 200, 3), (i,j))[-1]
            print("{0} {1} {2}".format(i, j, tuple(point)))
    '''

    while True:
        set_is_training = tf.assign(is_training, True)
        is_train = sess.run([set_is_training, is_training])[1]

        mini_batch_samples, mini_batch_labels = croper(images, labels, crop_per_image)
        plans = tl.image.pyramid_plan(crop_size, pyramid_scale)
        mini_batch_samples = tl.image.pyramid_apply(mini_batch_samples, plans)
        #debug_images_rects(sess, mini_batch_samples, plans, mini_batch_labels)


        mini_batch_samples = sess.run([mini_batch_samples])[0]
        model.run(sess, update_vars, feed_dict={input: mini_batch_samples})

        print("once finish")
        exit()


    sess.close()


if __name__ == "__main__":
    main()

