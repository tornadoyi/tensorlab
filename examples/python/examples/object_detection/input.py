import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.ops.geometry import rectangle_yx as rt, point_yx as pt
from ..support.utils import *


class Input(framework.Model):
    def __init__(self, sess, pyramid_scale):
        framework.Model.__init__(self)

        self._pyramid_scale = pyramid_scale
        self._pyramid_rate = (pyramid_scale - 1.0) / pyramid_scale

        self._input_images = tf.placeholder(tf.int32, [None, None, None, None])

        self._gen_net()


    @property
    def pyamid_rects_ratio_tensor(self): return self._pyramid_rects_ratio_tensor

    @property
    def pyramid_rects_tensor(self): return self._pyramid_rects_tensor

    @property
    def output_image_size_tensor(self): return self._output_image_size_tensor

    @property
    def output_shape_tensor(self): return self._output_shape_tensor

    @property
    def input_images(self): return self._input_images



    def _gen_net(self):

        self._input_shape_tensor = tf.shape(self._input_images)
        image_size_tensor = self._input_shape_tensor[1:3]

        # output
        self._output_image_size_tensor, \
        self._pyramid_rects_ratio_tensor = tl.image.pyramid_plan(image_size_tensor, self._pyramid_scale)
        self._output_shape_tensor = tf.convert_to_tensor([self._input_shape_tensor[0],
                                                  self._output_image_size_tensor[0],
                                                  self._output_image_size_tensor[1],
                                                  self._input_shape_tensor[3]])

        # pyramid rects
        pyramid_rects_tensor = rt.convert_ratio_to_value(self._pyramid_rects_ratio_tensor, tf.cast(self._output_image_size_tensor, tf.float32))
        self._pyramid_rects_tensor = tf.cast(pyramid_rects_tensor, tf.int32)

        # pyramid scales
        self._pyramid_scale_tensor = self._pyramid_rate ** tf.to_float(tf.range(0, tl.len(self._pyramid_rects_ratio_tensor), 1))

        # pyramid apply
        self.add(tl.image.pyramid_apply, self._input_images, self._output_image_size_tensor, self._pyramid_rects_ratio_tensor)



    def gen_rect_from_input_space_to_all_output_space(self, rects):
        rect_shape = tf.shape(rects)

        scales = self._pyramid_rate ** tf.to_float(tf.range(0, tl.len(self._pyramid_rects_ratio_tensor), 1))

        def transform_all_scales(s, trans_rects):
            i = s.step
            scale = self._pyramid_scale_tensor[i]
            scale_rects = self.gen_rect_from_input_space_to_output_space(rects, scale)
            return tf.concat([trans_rects, scale_rects], 0)

        trans_rects = tf.constant(0, tf.int32, (0, 4))
        trans_rects =  tl.for_loop(transform_all_scales, 0, tl.len(self._pyramid_scale_tensor),
                                   loop_vars=[trans_rects], auto_var_shape=True)

        return tf.reshape(trans_rects, [tl.len(scales), rect_shape[0], rect_shape[1]])


    def gen_rect_from_input_space_to_output_space(self, rects, scale):
        rects = tf.cast(rects, tf.int32)
        rect_count = tf.shape(rects)[0]
        tl_points = rt.top_left(rects)
        br_points = rt.bottom_right(rects)
        points = tf.concat([tl_points, br_points], 0)
        output_points = self.gen_point_from_input_space_to_output_space(points, scale)
        return rt.create(output_points[0:rect_count], output_points[rect_count:2*rect_count])



    def gen_point_from_input_space_to_output_space(self, points, scale):

        scale = tf.cast(scale, tf.float32)
        points = tf.cast(points, tf.int32)

        index = tf.cast(tf.log(scale) / tf.log(float(self._pyramid_rate)) + 0.5, tf.int32)
        index = tf.clip_by_value(index, 0, tf.shape(self._pyramid_rects_ratio_tensor)[0] - 1)

        target_scale = self._pyramid_rate ** tf.cast(index, tf.float32)
        target_scale = tf.reshape(target_scale, (-1, 1))
        target_scale = tf.concat((target_scale, target_scale), 1)   # scale_y, scale_x
        order = tf.reshape(tf.range(0, tf.shape(points)[0], 1), (-1, 1))
        order = tf.concat((order, tf.zeros([tf.shape(points)[0], 1], tf.int32)), 1)
        resize_space_points = tl.image.point_to_resize_space(points, target_scale, order, round_value=0.5)

        lr_points = rt.top_left(self._pyramid_rects_tensor)
        target_lr_points = tf.gather(lr_points, index)
        return target_lr_points + resize_space_points



    def gen_rect_from_output_space_to_input_space(self, rects):
        rects = tf.cast(rects, tf.int32)
        rect_count = tf.shape(rects)[0]
        points = rt.center(rects)


        indexes = self.gen_find_nearest_rects(points)
        target_rects = tf.gather(self._pyramid_rects_tensor, indexes)

        top_left = rt.top_left(target_rects)
        top_left_2 = tf.tile(top_left, (1, 2))

        relative_rects = rects - top_left_2
        relative_points = tf.reshape(relative_rects, [-1, 2])

        scales = 1.0 / (self._pyramid_rate ** tf.cast(indexes, tf.float32))
        scales = tf.tile(scales, [2])
        scales = tf.reshape(scales, (-1, 1))
        scales = tf.concat((scales, scales), 1)

        order = tf.reshape(tf.range(0, rect_count*2), (-1, 1))
        order = tf.concat([order, order], 1)

        resize_points = tl.image.point_to_resize_space(relative_points, scales, order, round_value=0.5)
        return tf.reshape(resize_points, (-1 ,4))



    def gen_point_from_output_space_to_input_space(self, points):
        points = tf.cast(points, tf.int32)
        point_count = tf.shape(points)[0]

        indexes = self.gen_find_nearest_rects(points)
        target_rects = tf.gather(self._pyramid_rects_tensor, indexes)


        # scale all points
        relative_points = points - rt.top_left(target_rects)

        scales = 1.0 / (self._pyramid_rate ** tf.cast(indexes, tf.float32))
        scales = tf.reshape(scales, (-1, 1))
        scales = tf.concat((scales, scales), 1)

        order = tf.reshape(tf.range(0, point_count), (-1, 1))
        order = tf.concat([order, order], 1)

        return tl.image.point_to_resize_space(relative_points, scales, order, round_value=0.5)



    def gen_find_nearest_rects(self, points):
        pyramid_rects = self._pyramid_rects_tensor

        rect_count = tf.shape(pyramid_rects)[0]
        point_count = tf.shape(points)[0]

        # find nearest rects
        ymin, ymax = rt.top(pyramid_rects), rt.bottom(pyramid_rects)
        xmin, xmax = rt.left(pyramid_rects), rt.right(pyramid_rects)
        min = tf.concat([tf.expand_dims(ymin, 1), tf.expand_dims(xmin, 1)], axis=1)
        max = tf.concat([tf.expand_dims(ymax, 1), tf.expand_dims(xmax, 1)], axis=1)
        min = tf.tile(min, (point_count, 1))
        max = tf.tile(max, (point_count, 1))

        points = tf.tile(points, [1, rect_count])
        points = tf.reshape(points, [-1, 2])

        nearest_points = tf.clip_by_value(points, min, max)
        vec = nearest_points - points
        length_square = pt.length_square(vec)
        length_square = tf.reshape(length_square, (point_count, -1))
        indexes = tf.argmin(length_square, 1)
        indexes = tf.cast(indexes, tf.int32)

        return indexes




    def debug_show(self, sess, croper, image_count):
        # get croper output
        crop_images, crop_rects, crop_splits = croper(sess, image_count, split_rects=False)

        # gen fetches
        pyramid_rects = self.gen_rect_from_input_space_to_all_output_space(tf.constant(crop_rects))

        results = sess.run([self.out, pyramid_rects], feed_dict={self._input_images: crop_images})

        # get all results
        output_images = results[0]
        scale_rects = results[1]
        splits = crop_splits

        pyramid_rect_list = []
        for rects in scale_rects:
            pyramid_rect_list.append(croper.split_rects(rects, splits))


        # show
        for i in xrange(len(output_images)):
            image = output_images[i].astype(np.uint8)

            # draw all rects
            for ratio_rect_list in pyramid_rect_list:
                for r in ratio_rect_list[i]:
                    y1, x1, y2, x2 = r
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow("image", image)
            print("progress {0}/{1}".format(i + 1, len(output_images)))
            press_key_stop()



    def test_rect_transform(self, sess):
        rect_count = 5
        idx = tf.random_uniform([], 0, tl.len(self._pyramid_rects_tensor), tf.int32)

        rect = self._pyramid_rects_tensor[idx]

        t = tf.random_uniform([1, rect_count], rect[0], rect[2], tf.int32)
        b = tf.random_uniform([1, rect_count], rect[0], rect[2], tf.int32)
        l = tf.random_uniform([1, rect_count], rect[1], rect[3], tf.int32)
        r = tf.random_uniform([1, rect_count], rect[1], rect[3], tf.int32)

        rects = tf.transpose(tf.concat([t, l, b, r], 0))

        scale = self._pyramid_rate ** tf.cast(idx, tf.float32)

        resize_rects = self.gen_rect_from_output_space_to_input_space(rects)
        origin_rects = self.gen_rect_from_input_space_to_output_space(resize_rects, scale)

        for i in xrange(100):
            rects_v, resize_rects_v, origin_rects_v, idx_v = sess.run([rects, resize_rects, origin_rects, idx ],
                                                                      feed_dict={self._input_images: np.zeros([4, 200, 200, 3])})

            equal = rects_v == origin_rects_v
            print("progress {0}/{1} idx:{2}".format(i+1, 100, idx_v))
            if equal.all(): continue

            print(rects_v)
            print("="*100)
            print(resize_rects_v)
            print("=" * 100)
            print(origin_rects_v)
            print("=" * 100)

            press_key_stop()



    def test_point_transform(self, sess):

        point_count = 5
        idx = tf.random_uniform([], 0, tl.len(self._pyramid_rects_tensor), tf.int32)

        rect = self._pyramid_rects_tensor[idx]

        y = tf.random_uniform([point_count], rect[0], rect[2], dtype=tf.int32)
        x = tf.random_uniform([point_count], rect[1], rect[3], dtype=tf.int32)
        points = pt.create(y, x)


        scale = self._pyramid_rate ** tf.cast(idx, tf.float32)

        resize_points = self.gen_point_from_output_space_to_input_space(points)
        origin_points = self.gen_point_from_input_space_to_output_space(resize_points, scale)

        for i in xrange(100):
            points_v, resize_points_v, origin_points_v, idx_v = sess.run([points, resize_points, origin_points ,idx],
                                                                         feed_dict={self._input_images: np.zeros([4, 200, 200, 3])})

            equal = points_v == origin_points_v
            print("progress {0}/{1} idx_v:{2}".format(i+1, 100, idx_v))
            if equal.all(): continue

            print(points_v)
            print("="*100)
            print(resize_points_v)
            print("=" * 100)
            print(origin_points_v)
            print("=" * 100)

            press_key_stop()
