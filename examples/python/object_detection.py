from __future__ import division

import math
import cv
import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.framework import layers
from tensorlab.runtime.geometry import rectangle_yx as rrt, point_yx as rpt
from tensorlab.ops.geometry import rectangle_yx as rt, point_yx as pt
from util import *
from support import dataset
from support.image import RandomCrop
import time

def load_data(file):
    def create_rectangle(t, l, w, h):
        return [t, l, t+h-1 , l+w-1]

    images, labels = dataset.load_object_detection_xml(file, create_rectangle)
    return images, labels



class Input(framework.Model):
    def __init__(self, sess, image_size, pyramid_scale):
        framework.Model.__init__(self)
        self._image_size = image_size
        self._pyramid_scale = pyramid_scale
        self._pyramid_rate = (pyramid_scale - 1.0) / pyramid_scale

        self._gen_pyramid_size_and_rects(sess)
        self._gen_net()
        self._gen_input_space_to_output_space()

    @property
    def pyamid_rects_ratio(self): return self._pyramid_rects_ratio

    @property
    def pyramid_rects(self): return self._pyramid_rects


    @property
    def output_size(self): return self._output_size


    @property
    def input(self): return self._input


    def input_space_to_output_space(self, sess, rects, scales):
        assert len(rects) == len(scales)

        points = rects.reshape((-1, 2))
        pt_scales = []
        for i in scales:
            pt_scales.append(i)
            pt_scales.append(i)

        resize_points = sess.run(self._input_to_output_space_tensor,
                 feed_dict={self._input_points: points, self._input_scale: pt_scales})

        return resize_points.reshape((-1, 4))


    def debug_show(self, sess, images, labels):
        pyramid_images = self.run(sess, feed_dict={self._input: images})[self.out]
        scales = self._pyramid_rate ** np.arange(0, len(self._pyramid_rects_ratio), 1)
        for i in xrange(len(labels)):
            image = pyramid_images[i]
            rects = labels[i]
            map_rects = []
            for j in xrange(len(self._pyramid_rects_ratio)):
                if len(rects) == 0:
                    new_rects = []
                else:
                    s = self._pyramid_rate ** j
                    new_rects = self.input_space_to_output_space(sess, rects, [s]*len(rects))
                map_rects.append(new_rects)


            for r in rects:
                y1, x1, y2, x2 = r
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=5)


            for r in self._pyramid_rects_ratio:
                y1, x1, y2, x2 = r
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            for rects in map_rects:
                for r in rects:
                    y1, x1, y2, x2 = r
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow("image", image)
            print("progress {0}/{1}".format(i + 1, len(pyramid_images)))
            press_key_stop()


    def _gen_pyramid_size_and_rects(self, sess):
        output_size_tensor, rect_tensor = tl.image.pyramid_plan(self._image_size, self._pyramid_scale)
        self._output_size, self._pyramid_rects_ratio, = sess.run([output_size_tensor, rect_tensor])
        self._pyramid_rects = rt.convert_ratio_to_size(self._pyramid_rects_ratio, self._output_size, dtype=np.int32)

        self._pyramid_rects_tensor = tf.convert_to_tensor(self._pyramid_rects)
        self._pyramid_rects_ratio_tensor = tf.convert_to_tensor(self._pyramid_rects_ratio)


    def _gen_net(self):
        self._input = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
        self.add(tl.image.pyramid_apply, self._input, self._output_size, self._pyramid_rects_ratio)



    def _gen_input_space_to_output_space(self):
        self._input_scale = tf.placeholder(dtype=tf.float32)
        self._input_points = tf.placeholder(dtype=tf.int32, shape=(None, 2))

        index_tensor = tf.cast(tf.log(self._input_scale ) / tf.log(float(self._pyramid_rate)) + 0.5, tf.int32)
        index_tensor = tf.clip_by_value(index_tensor, 0, tf.shape(self._pyramid_rects_ratio)[0] - 1)

        scale_tensor = self._pyramid_rate ** tf.cast(index_tensor, tf.float32)
        scale_tensor = tf.reshape(scale_tensor, (-1, 1))
        scale_tensor = tf.concat((scale_tensor, scale_tensor), 1)
        order_tensor = tf.reshape(tf.range(0, tf.shape(self._input_points)[0], 1), (-1, 1))
        order_tensor = tf.concat((order_tensor, order_tensor), 1)
        point_resize_space_tensor = tl.image.point_to_resize_space(self._input_points, scale_tensor, order_tensor)

        start_point = self._pyramid_rects[:,0:2]
        target_start_point_tensor = tf.gather(start_point, index_tensor)
        self._input_to_output_space_tensor =  target_start_point_tensor + point_resize_space_tensor


    def _gen_output_space_to_input_space(self, rect):
        def _nearest_rect(point):
            def _body(i, found, idx, best_dist):
                rect = self._pyramid_rects_tensor[i]

                def _contain():
                    found.assign(True)
                    return i

                def _not_contain():
                    dist = self._gen_nearest_point_square_dis(rect, point)
                    def _set_neart_point():
                        best_dist.assign(dist)
                        return i

                    return tf.cond(dist < best_dist, _set_neart_point, lambda : idx)

                temp_idx = tf.cond(trt.contains(rect, point), _contain, _not_contain)
                idx.assign(temp_idx)

                return i+1, found, idx, best_dist

            best_dist = tf.Variable(sys.maxint, trainable=False)
            idx = tf.Variable(0, trainable=False)
            found = tf.Variable(False, trainable=False)
            i = tf.Variable(0, dtype=tf.int32, trainable=False)

            i, found, idx, best_dist = tf.while_loop(
                lambda i, found: i < len(self._pyramid_rects) or found,
                _body, [i, found, idx, best_dist])

            return idx

        index_tensor = _nearest_rect(trt.center(rect))
        pyramid_rect = self._pyramid_rects_tensor[index_tensor]



    def _gen_nearest_point_square_dis(self, rect, point):
        temp = tf.Variable(point, dtype=point.dtype, trainable=False)
        temp[0].assign(tf.clip_by_value(rect[0], rect[2]))
        temp[1].assign(tf.clip_by_value(rect[1], rect[3]))

        return tf.reduce_sum(tf.square(temp - point))



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



class mmod_loss(object):
    def __init__(self,
                 input_layer,
                 model,
                 input_size,
                 detector_size,
                 loss_per_false_alarm = 1,
                 loss_per_missed_target = 1,
                 truth_match_iou_threshold = 0.5,):

        self._input_layer = input_layer
        self._model = model
        self._input_size = input_size
        self._detector_size = detector_size
        self._loss_per_false_alarm = loss_per_false_alarm
        self._loss_per_missed_target = loss_per_missed_target
        self._truth_match_iou_threshold = truth_match_iou_threshold


    def __call__(self):
        pass


    def _gen_loss(self, samples, labels):
        pass


    def _collect_all_box_from_output(self, sess, input_samples, input_labels, output_samples, adjust_threshold):
        input_shape = input_samples.shape
        batch, row, col, chanel = output_samples.shape
        assert chanel == 1

        def _collect(b, r, c, d):
            score = output_samples[b, r, c, 0]
            if score <= adjust_threshold: return
            output_point = pt.create(r, c)
            pramid_point = self._model.map_output_input(sess, input_shape, output_point)
            rect = rt.centered_rect(pramid_point, self._detector_size)
            #self._input_layer.


        self.loop_all_images(batch, row, col, chanel, _collect)


        for b in xrange(batch):
            for r in xrange(row):
                for c in xrange(col):
                    pass




    def loop_all_images(self, batch, row, col, chanel, call_back):
        for b in xrange(batch):
            for r in xrange(row):
                for c in xrange(col):
                    for d in xrange(chanel):
                        call_back(b, r, c, d)



def main():
    crop_size = (200, 200)
    crop_per_image = 30
    pyramid_scale = 6

    # create session
    sess = tf.InteractiveSession()

    # load train datas
    images, labels = load_data("../data/training.xml")

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

    croper = RandomCrop(images, labels, crop_size,
                        probability_use_label = 1.0,
                        max_roatation_angle=0,
                        translate_amount = 0,
                        random_scale_range = (1, 1),
                        probability_flip = 0)

    crop_images, crop_rect_list = croper(sess, 1)
    for i in xrange(len(crop_images)):
        image = crop_images[i].astype(np.uint8)
        rects = crop_rect_list[i]
        print(len(rects))
        for r in rects:
            y1, x1, y2, x2 = r
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=5)
        cv2.imshow("image", image)
        press_key_stop()
    exit()

    # create input layer
    input_layer = Input(sess, crop_size, pyramid_scale)

    # create model
    is_training = tf.Variable(False, dtype=tf.bool)
    model = Model(input_layer.out, is_training)

    # create loss
    loss = mmod_loss(input_layer, model, 40, 40)

    # init all variables
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    # train
    while True:
        time_tag()
        set_is_training = tf.assign(is_training, True)
        is_train = sess.run([set_is_training, is_training])[1]

        # crop image
        mini_batch_samples, mini_batch_labels = croper(images, labels, crop_per_image)
        mini_batch_samples = sess.run(mini_batch_samples)
        print("crop image cost {0}".format(time_tag()))


        # debug pyramid image
        #input_layer.debug_show(sess, mini_batch_samples, mini_batch_labels)

        result = model.run(sess, update_vars, feed_dict={input_layer.input: mini_batch_samples})
        print("run model cost {0}".format(time_tag()))

        output = result[model.out]
        print(output.shape)
        print(output)
        print("="*100)
        #print("once finish")
        #exit()


    sess.close()


if __name__ == "__main__":
    main()

