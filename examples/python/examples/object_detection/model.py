from __future__ import division

import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.framework import layers
from tensorlab.ops.geometry import rectangle_yx as rt, point_yx as pt
from ..support.utils import *


class Model(framework.Model):
    def __init__(self, input, is_training):
        framework.Model.__init__(self)
        self._input_layer = input
        self._gen_net(input.out, is_training)
        self._gen_output_shape_tensors()
        self._gen_padding_tensors()


    @property
    def input_layer(self): return self._input_layer


    @property
    def output_shape_tensor(self): return self._output_shape_tensors[-1]



    # modify dlib conv_ template
    # int _padding_y = _stride_y!=1? 0 : _nr/2 == 1 ? 1 : 2,
    # int _padding_x = _stride_x!=1? 0 : _nc/2 == 1 ? 1 : 2
    def gen_map_input_to_output_tensor(self, points):
        for i in xrange(len(self)):
            layer = self[i]
            if layer.core_name == "conv2d":
                padding = self._padding_tensors[i]
                filter = np.array(layer.core.filter[0:2], dtype=np.int32)
                strides = np.array(layer.core.strides[1:3], dtype=np.int32)
                points = (points + padding - tf.cast(filter / 2, tf.int32)) / strides
                points = tf.cast(points, tf.int32)

                #points = tl.Print(points, message="filter:{0} strides:{1} yx: ".format(filter, strides))


        #return points
        return pt.clip(points, [0, self.output_shape_tensor[1]], [0, self.output_shape_tensor[2]])



    def gen_map_output_to_input_tensor(self, points):
        for i in xrange(len(self) - 1, -1, -1):
            layer = self[i]
            if layer.core_name == "conv2d":
                padding = self._padding_tensors[i]
                filter = np.array(layer.core.filter[0:2], dtype=np.int)
                strides = np.array(layer.core.strides[1:3], dtype=np.int)
                points = points * strides - padding + tf.cast(filter / 2, tf.int32)

                # points = tl.Print(points, message="filter:{0} strides:{1} yx: ".format(filter, strides))

        #return points
        input_shape = self._input_layer.input_shape
        return pt.clip(points, [0, input_shape[1]], [0, input_shape[2]])



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
        self._output_shape_tensors = []
        input_shape = self._input_layer.output_shape_tensor
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
                input_shape = self._input_layer.output_shape_tensor
            else:
                input_shape = self._output_shape_tensors[i-1]
            input_shape = input_shape[1:3]

            if layer.core_name == "conv2d":
                padding = layer.core.padding_value(input_shape)
                self._padding_tensors.append(padding)
            else:
                self._padding_tensors.append(tf.constant((0,0), dtype=tf.int32))




    def test_map_points(self, sess):
        # test paddings
        conv_paddings = []
        for i in xrange(len(self)):
            layer = self[i]
            if layer.core_name == "conv2d":
                padding = self._padding_tensors[i]
                conv_paddings.append(padding)

        images = np.zeros(shape=[150, 200, 200, 3])


        points = []
        for y in xrange(100):
            for x in xrange(100):
                points.append([y, x])

        points = tf.constant(points)

        if False:
            map_points = self.gen_map_input_to_output_tensor(points)
        else:
            map_points = self.gen_map_output_to_input_tensor(points)


        feches = [map_points] + conv_paddings
        results = sess.run(feches, feed_dict={self._input_layer.input_images: images})

        map_points = results[0]
        paddings = results[1:len(feches)]

        for pad in paddings:
            print(pad)

        print("="* 100)
        for p in map_points:
            print(tuple(p))

        press_key_stop()