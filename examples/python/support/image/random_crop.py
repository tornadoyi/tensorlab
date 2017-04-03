import os
import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorlab.runtime.support import rectangle_yx as rt, point_yx as pt
import copy

CropPlan = namedtuple("CropPlan", ["rect", "flip", "angle"])


class RandomCrop(object):
    def __init__(self,
                 set_chip_dims,
                 probability_use_label = 1.0,
                 max_roatation_angle = 0,
                 translate_amount = 0.0,
                 random_scale_range=(1.0, 1.0),
                 probability_flip = 0,
                 min_rect_ratio = 0.025,
                 min_part_rect_ratio = 0.5,):

        self._chips_dims = np.array(set_chip_dims)
        self._random_scale_range = random_scale_range
        self._max_roatation_angle = max_roatation_angle
        self._probability_use_label = probability_use_label
        self._translate_amount = translate_amount
        self._probability_flip = probability_flip
        self._min_rect_ratio = min_rect_ratio
        self._min_part_rect_ratio = min_part_rect_ratio


    def __call__(self, images, labels, crops_per_image):
        assert len(images) == len(labels)

        # gen tensors and rects for all images
        crop_tensor_list = []
        crop_rects_list = []
        for i in xrange(len(images)):
            image = images[i]
            rects = labels[i]

            plans = []
            for i in xrange(crops_per_image):
                plan = self._make_plan(image.shape, rects)
                plans.append(plan)

            crop_tensor_list += self.gen_extract_image_chip_tensor(image, plans)
            crop_rects_list += self.gen_crop_image_rects(image.shape, rects, plans)

        # shuffle
        index = range(len(crop_tensor_list))
        np.random.shuffle(index)

        # concat tensor
        crop_tensor = None
        crop_rects = []
        for i in index:
            t = crop_tensor_list[i]
            crop_tensor = t if crop_tensor is None else tf.concat([crop_tensor, t], 0)
            crop_rects.append(crop_rects_list[i])

        return crop_tensor, crop_rects


    def gen_extract_image_chip_tensor(self, image, plans):
        r, c, d = image.shape
        images = np.array([image] * len(plans))
        images_tensor = tf.image.convert_image_dtype(images, tf.float32)

        # make boxes and transform
        box_ind = np.array(range(len(plans)), dtype=np.int32)
        transform = None
        boxes = []
        for plan in plans:
            rect = plan.rect
            center = rt.center(rect)
            # -angle because tensorflow transform inverse angle, i dont know why ..
            trans = self.gen_transform_mat(image.shape, center, angle=-plan.angle)
            transform = trans if transform is None else np.row_stack((transform, trans))
            boxes.append(rt.convert_size_to_ratio(rect, (r, c)))
        boxes = np.array(boxes)


        # rotate translate
        trans_images_tensor = tf.contrib.image.transform(images_tensor, transform)

        # crop
        crops_tensor = tf.image.crop_and_resize(trans_images_tensor, boxes=boxes, box_ind=box_ind, crop_size=self._chips_dims)

        # flip
        chip_r, chip_c = self._chips_dims
        flip_tensors = []
        for i in xrange(len(plans)):
            plan = plans[i]
            tensor = crops_tensor[i]
            if plan.flip:
                tensor = tf.image.flip_left_right(tensor)
            tensor = tf.reshape(tensor, (1, chip_r, chip_c, d))
            flip_tensors.append(tensor)
        return flip_tensors


    def gen_crop_image_rects(self, image_shape, rects, plans):
        r, c, d = image_shape
        image_rect = rt.create_with_size((0, 0), (r, c))
        min_object_size = self._chips_dims[0] * self._chips_dims[1] * self._min_rect_ratio

        transform_list = []
        scale_list = []
        left_top_list = []
        for plan in plans:
            rect = plan.rect
            center = rt.center(rect)
            scale = self._chips_dims / rt.size(rect)
            trans = self.gen_transform_mat(image_shape, center, angle=plan.angle)
            transform_list.append(trans)
            scale_list.append(scale)
            left_top_list.append(rt.top_left(rect))


        rects_list = []
        for i in xrange(len(transform_list)):
            plan = plans[i]
            trans = transform_list[i]
            scale = scale_list[i]
            left_top = left_top_list[i]
            trans = trans[0]

            rects_array = None
            for rect in rects:
                center = rt.center(rect)
                trans_center = self.apply_transoform(center, trans)
                reltive_scale_center = (trans_center - left_top) * scale
                if plan.flip: reltive_scale_center[1] = self._chips_dims[1] - reltive_scale_center[1]
                trans_rect = rt.centered_rect(reltive_scale_center, rt.size(rect)*scale)

                # filter rect not in crop
                if rt.area(trans_rect) < min_object_size: continue

                trans_rect_width, trans_rect_height = rt.width(trans_rect), rt.height(trans_rect)
                clip_rect = rt.clip_left_right(trans_rect, 0, self._chips_dims[1] - 1)
                clip_rect = rt.clip_top_bottom(clip_rect, 0, self._chips_dims[0] - 1)
                final_rect = clip_rect.astype(np.int)

                # filter rect not or part in crop
                if rt.area(final_rect) < min_object_size or \
                    rt.width(final_rect)  < trans_rect_width * self._min_part_rect_ratio or \
                    rt.height(final_rect) < trans_rect_height * self._min_part_rect_ratio:
                    continue
                #rects_list[i].append(final_rect)
                final_rect = final_rect.reshape((1, len(final_rect)))
                rects_array = final_rect if rects_array is None else np.concatenate((rects_array, final_rect), 0)

            if rects_array is None:
                rects_list.append(np.array([]))
            else:
                rects_list.append(rects_array)

        return rects_list



    def _make_plan(self, image_shape, rects):
        r, c, d = image_shape
        should_flip = np.random.uniform(0, 1) > self._probability_flip
        angle = np.random.uniform(-self._max_roatation_angle, self._max_roatation_angle)

        # get rect from labels
        plan = None
        rand_rect = None
        if np.random.uniform(0, 1) < self._probability_use_label:
            index = np.random.randint(0, len(rects))
            rect = rects[index]
            size = np.min((rt.width(rect), rt.height(rect)))
            center = rt.center(rect)

            rand_translate = (np.random.uniform(-self._translate_amount, self._translate_amount) * rt.height(rect),
                              np.random.uniform(-self._translate_amount, self._translate_amount) * rt.width(rect))


            scale = np.random.uniform(*self._random_scale_range)


            scale_size = size * scale
            offset_center = center + rand_translate
            rand_rect = rt.centered_rect(offset_center, (scale_size, scale_size))

        else:
            scale = np.random.uniform(0.1, 0.95)
            size = scale * np.min((r, c))
            point = pt.create(np.random.randint(0, 65535) % (r - size),
                      np.random.randint(0, 65535) % (c - size))

            rand_rect = rt.create_with_size(point, (size, size))

        return CropPlan(rand_rect, should_flip, angle)



    def gen_transform_mat(self, image_shape, anchor, translate=[0, 0], angle=0, scale=[1, 1]):
        anchor = np.array([anchor[1], anchor[0]], dtype=np.float32)
        translate = np.array([translate[1], translate[0]], dtype=np.float32)
        scale = np.array([scale[1], scale[0]], dtype=np.float32)

        r, c, d = image_shape
        radian = np.deg2rad(angle)
        r_mat = np.array([[np.cos(radian), np.sin(radian)],
                          [-np.sin(radian), np.cos(radian)]], dtype=np.float32)

        r_mat[0, 0] /= scale[0]
        r_mat[1, 1] /= scale[1]

        t_vec = np.dot(r_mat, anchor * -1) + anchor + translate
        mat = np.column_stack((r_mat, t_vec))
        mat = np.array([list(mat[0]) + list(mat[1]) + [0] * 2], dtype=np.float32)

        return mat

    def apply_transoform(self, point, trans):
        point = point[::-1]
        r_mat = np.array([trans[[0, 1]], trans[[3, 4]]])
        t_vec = trans[[2, 5]]
        result = np.dot(r_mat, point) + t_vec
        return result[::-1]