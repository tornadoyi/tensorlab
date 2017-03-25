import os
import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorlab import Rectanglef, Point2f
import copy

CropPlan = namedtuple("CropPlan", ["rect", "flip", "angle"])


class RandomCrop(object):
    def __init__(self,
                 set_chip_dims,
                 probability_use_label = 0.5,
                 max_roatation_angle = 30,
                 translate_amount = 0.1,
                 random_scale_range=(1.3, 4),
                 probability_flip = 0.5,
                 min_rect_ratio = 0.01,#0.025,
                 min_part_rect_ratio = 0.4,):
        self._chips_dims = set_chip_dims
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

        # shuffle and concat tensor
        index = range(len(crop_tensor_list))
        np.random.shuffle(index)
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
            center = rect.center
            # -angle because tensorflow transform inverse angle, i dont know why ..
            trans = self.gen_transform_mat(image.shape, (center[0], center[1]), angle=-plan.angle)
            if transform is None:
                transform = trans
            else:
                transform = np.row_stack((transform, trans))
            boxes.append([rect.top/(r-1), rect.left/(c-1), rect.bottom/(r-1), rect.right/(c-1)])


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
        image_rect = Rectanglef.create_with_tlwh(Point2f(0, 0), c, r)
        min_object_size = self._chips_dims[0] * self._chips_dims[1] * self._min_rect_ratio

        transform_list = []
        scale_list = []
        left_top_list = []
        for plan in plans:
            rect = plan.rect
            center = rect.center
            scale = [self._chips_dims[0]/rect.width, self._chips_dims[1]/rect.height]
            trans = self.gen_transform_mat(image_shape, (center[0], center[1]), angle=plan.angle)
            transform_list.append(trans)
            scale_list.append(scale)
            left_top_list.append(np.array([rect.left, rect.top]))


        rects_list = []
        for i in xrange(len(transform_list)):
            plan = plans[i]
            rects_list.append([])
            trans = transform_list[i]
            scale = scale_list[i]
            left_top = left_top_list[i]
            trans = trans[0]
            r_mat = np.array([trans[[0,1]], trans[[3,4]]])
            t_vec = trans[[2,5]]
            for rect in rects:
                center = rect.center
                center = np.array([center[0], center[1]])
                trans_center = np.dot(r_mat, center) + t_vec
                reltive_scale_center = (trans_center - left_top) * scale
                if plan.flip: reltive_scale_center[0] = self._chips_dims[0] - reltive_scale_center[0]
                trans_rect = Rectanglef.create_with_center(Point2f(reltive_scale_center[0], reltive_scale_center[1]),
                                                           rect.width*scale[0], rect.height*scale[1])

                # filter rect not in crop
                if trans_rect.area < min_object_size: continue

                trans_rect_width, trans_rect_height = trans_rect.width, trans_rect.height
                clip_rect = trans_rect
                clip_rect.clip_left_right(0, self._chips_dims[0] - 1)
                clip_rect.clip_top_bottom(0, self._chips_dims[1] - 1)

                # filter rect not or part in crop
                if clip_rect.area < min_object_size or \
                    clip_rect.width  < trans_rect_width * self._min_part_rect_ratio or \
                    clip_rect.height  < trans_rect_height * self._min_part_rect_ratio:
                    continue
                rects_list[i].append(clip_rect)

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
            size = np.min((rect.width, rect.height))
            center = rect.center

            rand_translate = (np.random.uniform(-self._translate_amount, self._translate_amount) * rect.width,
                              np.random.uniform(-self._translate_amount, self._translate_amount) * rect.height)


            scale = np.random.uniform(*self._random_scale_range)


            scale_size = size * scale
            offset_center = Point2f(center[0] + rand_translate[0], center[1] + rand_translate[1])
            rand_rect = Rectanglef.create_with_center(offset_center, scale_size, scale_size)

        else:
            scale = np.random.uniform(0.1, 0.95)
            size = scale * np.min((r, c))
            rand_xy = (np.random.randint(0, 65535) % (c - size),
                      np.random.randint(0, 65535) % (r - size))
            point = Point2f(float(rand_xy[0]), float(rand_xy[1]))

            rand_rect = Rectanglef.create_with_tlwh(point, size, size)

        return CropPlan(rand_rect, should_flip, angle)
        #return CropPlan(rand_rect, False, 0)


    def gen_transform_mat(self, image_shape, anchor, translate=[0, 0], angle=0, scale=[1, 1]):
        anchor = np.array(anchor, dtype=np.float32)
        translate = np.array(translate, dtype=np.float32)
        scale = np.array(scale, dtype=np.float32)

        r, c, d = image_shape
        radian = np.deg2rad(angle)
        r_mat = np.array([[np.cos(radian), np.sin(radian)],
                          [-np.sin(radian), np.cos(radian)]], dtype=np.float32)

        r_mat[0, 0] /= scale[0]
        r_mat[1, 1] /= scale[1]

        t_vec = np.dot(r_mat, anchor * -1) + anchor + np.array(translate)
        mat = np.column_stack((r_mat, t_vec))
        mat = np.array([list(mat[0]) + list(mat[1]) + [0] * 2], dtype=np.float32)

        return mat




