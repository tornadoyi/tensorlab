import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.ops.geometry import rectangle_yx as rt
import cv2
from ..support.utils import *


class Input(framework.Model):
    def __init__(self, sess, croper, image_size, pyramid_scale):
        framework.Model.__init__(self)
        self._croper = croper
        self._image_size = image_size
        self._pyramid_scale = pyramid_scale
        self._pyramid_rate = (pyramid_scale - 1.0) / pyramid_scale

        self._gen_pyramid_size_and_rects(sess)
        self._gen_net()



    @property
    def pyamid_rects_ratio(self): return self._pyramid_rects_ratio

    @property
    def pyramid_rects(self): return self._pyramid_rects

    @property
    def output_size(self): return self._output_size



    def _gen_pyramid_size_and_rects(self, sess):
        # gen tensors
        output_size_tensor, pyramid_rects_ratio_tensor = tl.image.pyramid_plan(self._image_size, self._pyramid_scale)
        pyramid_rects_tensor = rt.convert_ratio_to_value(pyramid_rects_ratio_tensor, tf.cast(output_size_tensor, tf.float32))
        pyramid_rects_tensor = tf.cast(pyramid_rects_tensor, tf.int32)

        # run tensors
        self._output_size, \
            self._pyramid_rects, \
            self._pyramid_rects_ratio= sess.run([output_size_tensor, pyramid_rects_tensor, pyramid_rects_ratio_tensor])

        # convert to tensors
        self._pyramid_rects_tensor = tf.convert_to_tensor(self._pyramid_rects)
        self._pyramid_rects_ratio_tensor = tf.convert_to_tensor(self._pyramid_rects_ratio)


    def _gen_net(self):
        crop_image_tensor, _, _ = self._croper.output_tensors
        self.add(tl.image.pyramid_apply, crop_image_tensor, self._output_size, self._pyramid_rects_ratio)



    def _gen_rect_from_input_space_to_output_space(self, rects, scale):
        rects = tf.cast(rects, tf.int32)
        rect_count = tf.shape(rects)[0]
        tl_points = rt.top_left(rects)
        br_points = rt.bottom_right(rects)
        points = tf.concat([tl_points, br_points], 0)
        output_points = self._gen_point_from_input_space_to_output_space(points, scale)
        return rt.create(output_points[0:rect_count], output_points[rect_count:2*rect_count])


    def _gen_point_from_input_space_to_output_space(self, points, scale):
        scale = tf.cast(scale, tf.float32)
        points = tf.cast(points, tf.int32)

        index = tf.cast(tf.log(scale) / tf.log(float(self._pyramid_rate)) + 0.5, tf.int32)
        index = tf.clip_by_value(index, 0, tf.shape(self._pyramid_rects_ratio)[0] - 1)

        target_scale = self._pyramid_rate ** tf.cast(index, tf.float32)
        target_scale = tf.reshape(target_scale, (-1, 1))
        target_scale = tf.concat((target_scale, target_scale), 1)   # scale_y, scale_x
        order = tf.reshape(tf.range(0, tf.shape(points)[0], 1), (-1, 1))
        order = tf.concat((order, tf.zeros([tf.shape(points)[0], 1], tf.int32)), 1)
        resize_space_points = tl.image.point_to_resize_space(points, target_scale, order)

        lr_points = rt.top_left(self._pyramid_rects_tensor)
        target_lr_points = tf.gather(lr_points, index)
        return target_lr_points + resize_space_points



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



    def debug_show(self, sess, image_count):
        # get croper output
        crop_images, crop_rects, crop_splits = self._croper.output_tensors

        # gen fetches
        scales = self._pyramid_rate ** np.arange(0, len(self._pyramid_rects_ratio), 1)
        pyramid_rect_list = []
        for s in scales:
            pyramid_rects = self._gen_rect_from_input_space_to_output_space(crop_rects, s)
            pyramid_rect_list.append(pyramid_rects)

        fetches = [self.out, crop_splits] + pyramid_rect_list
        results = sess.run(fetches, self._croper.gen_feed_dict(image_count))

        # get all results
        output_images = results[0]
        splits = results[1]
        pyramid_rect_list = []
        for i in xrange(2, len(results), 1):
            rects = results[i]
            pyramid_rect_list.append(self._croper.split_rects(rects, splits))


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

