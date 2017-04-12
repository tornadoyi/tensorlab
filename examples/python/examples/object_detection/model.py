import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.framework import layers



class Model(framework.Model):
    def __init__(self, input, is_training):
        framework.Model.__init__(self)
        self._input_layer = input
        self._gen_net(is_training)
        self._gen_output_shape_tensors()
        self._gen_padding_tensors()
        self._gen_map_input_to_output_tensor()
        self._gen_map_output_to_input_tensor()


    @property
    def input_layer(self): return self._input_layer

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


    def _gen_net(self, is_training):
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

        input, _, _ = self._input_layer.output_tensor

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
