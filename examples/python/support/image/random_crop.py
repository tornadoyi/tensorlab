import os
import copy
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorlab as tl
from tensorlab.ops.geometry import rectangle_yx as rt, point_yx as pt


class RandomCrop(object):

    def __init__(self,
                 image_list,
                 label_list,
                 set_chip_dims,
                 probability_use_label = 1.0,
                 max_roatation_angle = 0,
                 translate_amount = 0.0,
                 random_scale_range=(1.0, 1.0),
                 probability_flip = 0,
                 min_rect_ratio = 0.025,
                 min_part_rect_ratio = 0.5,):

        self._image_list = image_list
        self._label_list = label_list

        self._chips_dims = np.array(set_chip_dims)
        self._random_scale_range = random_scale_range
        self._max_roatation_angle = max_roatation_angle
        self._probability_use_label = probability_use_label
        self._translate_amount = translate_amount
        self._probability_flip = probability_flip
        self._min_rect_ratio = min_rect_ratio
        self._min_part_rect_ratio = min_part_rect_ratio

        self._gen_init_image_rect_tensor()
        self._gen_random_crop_tensor()


    def __call__(self, sess, crop_count):

        indexes = np.random.uniform(0, len(self._image_list), crop_count).astype(np.int32)
        index_dict = {}
        for i in indexes:
            if not index_dict.has_key(i):
                index_dict[i] = 0
            index_dict[i] += 1

        input_indexes = index_dict.items()

        (crop_images, crop_rects, crop_splits) = sess.run(
            [self._crop_image_tensor, self._crop_rect_tensor, self._crop_split_tensor],
            feed_dict= {self._input_gen_indexes: input_indexes})

        return crop_images


    def _gen_init_image_rect_tensor(self):
        self._images_array = tf.TensorArray(tf.float32, size=len(self._image_list), clear_after_read=False, infer_shape=False)
        self._labels_array = tf.TensorArray(tf.float32, size=len(self._label_list), clear_after_read=False, infer_shape=False)

        image_shapes = np.zeros((0, 3), np.int32)
        for i in xrange(len(self._image_list)):
            img = self._image_list[i]
            self._images_array = self._images_array.write(i, tf.cast(img, tf.float32))
            image_shapes = np.concatenate([image_shapes, np.expand_dims(img.shape, 0)], 0)

        label_shapes = np.zeros((0, 2), np.int32)
        for i in xrange(len(self._label_list)):
            lab = np.array(self._label_list[i], np.float32)
            self._labels_array = self._labels_array.write(i, lab)
            label_shapes = np.concatenate([label_shapes, np.expand_dims(lab.shape, 0)], 0)


        self._image_shapes_tensor = tf.constant(image_shapes, tf.int32)
        self._label_shapes_tensor = tf.constant(label_shapes, tf.int32)


    def _gen_random_crop_tensor(self):
        self._input_gen_indexes = tf.placeholder(tf.int32, (None, 2))
        gen_image_count = tf.shape(self._input_gen_indexes)[0]

        # random crop image
        def _random_crop(s, crop_images, crop_rects, crop_splits, cur_gen_count):
            i = s.step
            index, count = self._input_gen_indexes[i, 0], self._input_gen_indexes[i, 1]

            def _crop(crop_images, crop_rects, crop_splits, cur_gen_count):
                img = self._images_array.read(index)
                image = tf.reshape(self._images_array.read(index), self._image_shapes_tensor[index])
                rects = tf.reshape(self._labels_array.read(index), self._label_shapes_tensor[index])

                image_shape = tf.shape(image)

                # 3 steps generate image crops
                plans = self._gen_make_plans_tensor(image_shape, rects, count)
                chip_images = self._gen_extract_image_chip_tensor(image, *plans)
                chip_rects, rect_splits = self._gen_crop_image_rects_tensor(image_shape, rects, *plans)

                # 4 save
                crop_images = tf.concat([crop_images, chip_images], 0)
                crop_rect = tf.concat([crop_rects, chip_rects], 0)
                crop_splits = tf.concat([crop_splits, rect_splits + cur_gen_count], 0)

                cur_gen_count = cur_gen_count + count
                return crop_images, crop_rects, crop_splits, cur_gen_count

            return tf.cond(tf.equal(count, 0),
                           lambda: (crop_images, crop_rects, crop_splits, cur_gen_count),
                           lambda: _crop(crop_images, crop_rects, crop_splits, cur_gen_count))

        crop_images = tf.constant(0, tf.float32, (0, self._chips_dims[0], self._chips_dims[1], 3))
        crop_rects = tf.constant(0, tf.int32, (0, 4))
        crop_splits = tf.constant(0, tf.int32, (0, 2))
        cur_gen_count = tf.constant(0, tf.int32)
        crop_images, crop_rects, crop_splits, cur_gen_count = tl.for_loop(_random_crop, 0, gen_image_count,
                    loop_vars=[crop_images, crop_rects, crop_splits, cur_gen_count], auto_var_shape=True)

        # shuffle
        '''
        indexes = tf.random_shuffle(tf.range(0, self._input_gen_count))
        crop_images = tf.gather(crop_images, indexes)

        def _rect_shuffle(s, shuffle_rect_array):
            i = s.step
            index = indexes[i]
            return shuffle_rect_array.write(i, crop_rect_array.read(index))

        shuffle_rect_array = tf.TensorArray(tf.int32, self._input_gen_count, clear_after_read=False, infer_shape=False)
        crop_rect_array = tl.for_loop(_rect_shuffle, 0, self._input_gen_count, loop_vars=[shuffle_rect_array])
        '''

        self._crop_image_tensor, self._crop_rect_tensor, self._crop_split_tensor = crop_images, crop_rects, crop_splits


    def _gen_extract_image_chip_tensor(self, image, chip_rects, flips, angles):
        chip_count = tf.shape(chip_rects)[0]

        images = tf.tile(tf.expand_dims(image, 0), (chip_count, 1, 1, 1))
        image_shape = tf.shape(image)
        r, c, d = image_shape[0], image_shape[1], image_shape[2]
        box_ind = tf.range(0, chip_count)

        def _gen_transform(s, transform, boxes):
            i = s.step
            rect = chip_rects[i]
            angle = angles[i]
            center = rt.center(rect)

            # mat for transform
            trans_mat = self._gen_transform_mat(image_shape, center, angle=angle)
            trans_mat = tf.expand_dims(trans_mat, 0)
            transform = tf.concat([transform, trans_mat], 0)

            # box
            box = rt.convert_value_to_ratio(rect, pt.create(r, c, tf.float32))
            box = tf.expand_dims(box, 0)
            boxes = tf.concat([boxes, box], 0)

            return transform, boxes

        # gen all transform
        transform, boxes = tf.constant(0, tf.float32, (0, 8)), tf.constant(0, tf.float32, (0, 4))
        transform, boxes = tl.for_loop(_gen_transform, 0, chip_count, loop_vars=[transform, boxes], auto_var_shape=True)

        # apply transform
        trans_images = tf.contrib.image.transform(images, transform)

        # crop and resize
        crop_image = tf.image.crop_and_resize(images, boxes=boxes, box_ind=box_ind, crop_size=self._chips_dims)

        # flip
        def _flip_images(s, flip_images):
            i = s.step
            image = crop_image[i]
            flip = flips[i]
            image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
            image = tf.expand_dims(image, 0)
            return tf.concat([flip_images, image], 0)

        flip_images = tl.fill((0, self._chips_dims[0], self._chips_dims[1], d), 0, tf.float32)
        return tl.for_loop(_flip_images, 0, chip_count, loop_vars=[flip_images], auto_var_shape=True)



    def _gen_crop_image_rects_tensor(self, image_shape, origin_rects, chip_rects, flips, angles):
        chip_dims = tf.cast(self._chips_dims, dtype=tf.float32)
        chip_shape = tf.shape(chip_rects)
        origin_rect_shape = tf.shape(origin_rects)
        chip_count, chip_rect_dim = chip_shape[0], chip_shape[1]
        origin_rect_count = origin_rect_shape[0]
        r, c, d = image_shape[0], image_shape[1], image_shape[2]
        min_object_size = chip_dims[0] * chip_dims[1] * self._min_rect_ratio


        def _map_rect_to_crop(s, map_rects, rect_splits, map_rect_count):
            i = s.step
            chip_rect = chip_rects[i]
            angle = angles[i]
            flip = flips[i]
            center = rt.center(chip_rect)
            scale = chip_dims / rt.size(chip_rect)
            trans = self._gen_transform_mat(image_shape, center, angle=-angle)
            top_left = rt.top_left(chip_rect)

            def _transform_rect(s, map_rects, map_rect_count):
                j = s.step
                rect = origin_rects[j]
                center = rt.center(rect)
                trans_center = self.apply_transoform(center, trans)
                reltive_scale_center = (trans_center - top_left) * scale

                relative_center = tf.cond(flip,
                        lambda: tf.convert_to_tensor([reltive_scale_center[0], chip_dims[1] - reltive_scale_center[0]]),
                        lambda: reltive_scale_center)

                trans_rect = rt.centered_rect(relative_center, rt.size(rect) * scale)


                # normalize final rect
                trans_rect_width, trans_rect_height = rt.width(trans_rect), rt.height(trans_rect)
                clip_rect = rt.clip_left_right(trans_rect, 0, chip_dims[1] - 1)
                clip_rect = rt.clip_top_bottom(clip_rect, 0, chip_dims[0] - 1)
                final_rect = tf.cast(clip_rect, tf.int32)


                # filter rect not or part in crop
                final_height, final_width = rt.height(final_rect), rt.width(final_rect)
                drop_cond = tl.logical_or(
                    tf.less(rt.area(trans_rect),  min_object_size),
                    tf.less(rt.area(final_rect), tf.cast(min_object_size, tf.int32)),
                    tf.less(final_width, tf.cast(trans_rect_width * self._min_part_rect_ratio, tf.int32)),
                    tf.less(final_height, tf.cast(trans_rect_height * self._min_part_rect_ratio, tf.int32)))


                def _save(map_rects):
                    normal_rect = tf.expand_dims(final_rect, 0)
                    map_rects = tf.concat([map_rects, normal_rect], 0)
                    return map_rects, map_rect_count+1

                return tf.cond(drop_cond, lambda: (map_rects, map_rect_count), lambda: _save(map_rects))


            # loop for map all origin rects to chip rect
            st = map_rect_count
            map_rects, map_rect_count = tl.for_loop(_transform_rect, 0, origin_rect_count,
                               loop_vars=[map_rects, map_rect_count], auto_var_shape=True)


            # update rect_splits
            split = tf.convert_to_tensor([[st, map_rect_count]])
            rect_splits = tf.concat([rect_splits, split], 0)

            return map_rects, rect_splits, map_rect_count


        map_rects = tl.fill((0, 4), 0, tf.int32)
        rect_splits = tl.fill((0, 2), 0, tf.int32)
        map_rect_count = tf.constant(0, tf.int32)
        map_rects, rect_splits, map_rect_count = tl.for_loop(_map_rect_to_crop, 0, chip_count,
                                            loop_vars=[map_rects, rect_splits, map_rect_count], auto_var_shape=True)

        return map_rects, rect_splits



    def _gen_make_plans_tensor(self, image_shape, rects, count):

        def _body(s, chip_rects, flips, angles):
            i = s.step
            rect, flip, angle = self._gen_make_plan_tensor(image_shape, rects)

            rect = tf.expand_dims(rect, 0)
            flip = tf.expand_dims(flip, 0)
            angle = tf.expand_dims(angle, 0)

            chip_rects = tf.concat([chip_rects, rect], 0)
            flips = tf.concat([flips, flip], 0)
            angles = tf.concat([angles, angle], 0)

            return chip_rects, flips, angles

        chip_rects = tf.constant(0, tf.float32, (0, 4))
        flips = tf.constant(0, tf.bool, (0,))
        angles = tf.constant(0, tf.float32, (0,))
        return tl.for_loop(_body, 0, count,  loop_vars=[chip_rects, flips, angles], auto_var_shape=True)



    def _gen_make_plan_tensor(self, image_shape, rects):
        # t_image_shape must be a vector with 3 elements r, c, d
        # t_rects shape must be n x 4

        assert image_shape.shape.ndims == 1 and image_shape.shape[0].value == 3
        #assert len(rects.shape) == 2 and rects.shape[1].value == 4

        r, c, d = image_shape[0], image_shape[1], image_shape[2]
        rect_count = tf.shape(rects)[0]

        should_flip = tf.random_uniform([], 0, 1) > self._probability_flip
        angle = tf.random_uniform([], -self._max_roatation_angle, self._max_roatation_angle)

        def _gen_by_rect():
            index = tf.random_uniform([], 0, rect_count, dtype=tf.int32)
            rect = rects[index]
            size = tf.minimum(rt.height(rect), rt.width(rect))
            center = rt.center(rect)

            rand_translate = tf.random_uniform([2], -self._translate_amount, self._translate_amount) * rt.size(rect)
            scale = tf.random_uniform([], *self._random_scale_range)

            scale_size = size * scale
            offset_center = center + rand_translate
            rand_rect = rt.centered_rect(offset_center, pt.create(scale_size, scale_size))
            return rand_rect


        def _gen_by_random():
            scale = tf.random_uniform([], 0.1, 0.95)
            f_size = scale * tf.cast(tf.minimum(r, c), tf.float32)
            size = tf.cast(f_size, tf.int32)
            point = pt.create(tf.random_uniform([], 0, 65535, dtype=tf.int32) % (r - size),
                              tf.random_uniform([], 0, 65535, dtype=tf.int32) % (c - size))

            rand_rect = rt.create_with_size(point, (size, size))
            return tf.cast(rand_rect, tf.float32)


        rand_rect= tf.cond(tf.random_uniform([], 0, 1) < tf.constant(self._probability_use_label, tf.float32),
                _gen_by_rect, _gen_by_random)


        return rand_rect, should_flip, angle


    def _gen_transform_mat(self, image_shape, anchor, translate=None, angle=None, scale=None):
        # t_image_shape must be a vector with 3 elements r, c, d

        # reverse all parameters
        anchor = tf.reverse(tf.convert_to_tensor(anchor, dtype=tf.float32), [0])

        if translate is None:
            translate = tf.constant([0.0, 0.0], dtype=tf.float32)
        else:
            translate = tf.reverse(tf.convert_to_tensor(translate, dtype=tf.float32), [0])

        if scale is None:
            scale = tf.constant([1.0, 1.0])
        else:
            scale = tf.reverse(tf.convert_to_tensor(scale, tf.float32), [0])

        if angle is None:
            angle = tf.constant(0.0, dtype=tf.float32)
        else:
            angle = tf.convert_to_tensor(angle, dtype=tf.float32)

        # flatten all parameters
        anchor = tl.flatten(anchor, column=True)
        translate = tl.flatten(translate, column=True)
        scale = tl.flatten(scale, column=True)

        r, c, d = image_shape[0], image_shape[1], image_shape[2]
        radian = tl.deg2rad(angle)

        r_mat = tf.convert_to_tensor(
            [[tf.cos(radian), -tf.sin(radian)],
             [tf.sin(radian), tf.cos(radian)]], dtype=tf.float32)

        s_mat = tf.convert_to_tensor(
            [[scale[0, 0], 1.0],
             [1.0, scale[1, 0]]], dtype=tf.float32)

        r_mat = r_mat / s_mat
        t_vec = -tf.matmul(r_mat, anchor) + anchor + translate

        mat = tf.concat([r_mat, t_vec], axis=1)
        mat = tf.concat([tl.flatten(mat), [0, 0]], 0)
        return mat



    def apply_transoform(self, point, trans):
        point = tf.reverse(point, [0])
        r_mat = tf.convert_to_tensor([tf.gather(trans, [0, 1]), tf.gather(trans, [3,4])])
        t_vec = tf.gather(trans, [2, 5])
        trans_point = tf.matmul(r_mat, tf.expand_dims(point, 0), transpose_b=True)
        result = tl.flatten(trans_point) + t_vec
        return tf.reverse(result, [0])









