import os
import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorlab import Rectanglef, Point2f

CropPlan = namedtuple("CropPlan", ["rect", "flip", "angle"])


class RandomCrop(object):
    def __init__(self,
                 set_chip_dims,
                 width_scale_range=(0.8, 1.3),
                 height_scale_range = (0.8, 1.3),
                 max_roatation_angle = 30,
                 probability_use_label = 1,#0.5,
                 translate_amount = 0.1,
                 probability_flip = 0.5,):
        self._chips_dims = set_chip_dims
        self._width_scale_range = width_scale_range
        self._height_scale_range = height_scale_range
        self._max_roatation_angle = max_roatation_angle
        self._probability_use_label = probability_use_label
        self._translate_amount = translate_amount
        self._probability_flip = probability_flip



    def __call__(self, images, labels, crops_per_image):
        image = images[0]
        rects = labels[0]

        plans = []
        for i in xrange(crops_per_image):
            plan = self._make_plan(image.shape, rects)
            plans.append(plan)

        return self.gen_extract_image_chip_tensor(image, plans)


    def gen_extract_image_chip_tensor(self, image, plans):
        r, c, d = image.shape
        box_ind = np.array(range(len(plans)), dtype=np.int32)
        transform = None
        boxes = []
        for plan in plans:
            rect = plan.rect
            center = rect.center
            trans = self.gen_transform_mat(image.shape, (center[0], center[1]), angle=plan.angle)
            if transform is None:
                transform = trans
            else:
                transform = np.row_stack((transform, trans))
            boxes.append([rect.top/(r-1), rect.left/(c-1), rect.bottom/(r-1), rect.right/(c-1)])

        images = np.array([image]*len(plans))
        images_tensor = tf.image.convert_image_dtype(images, tf.float32)

        trans_images_tensor = tf.contrib.image.transform(images_tensor, transform)
        crops_tensor = tf.image.crop_and_resize(images_tensor, boxes=boxes, box_ind=box_ind, crop_size=self._chips_dims)

        chip_r, chip_c = self._chips_dims
        flip_tensor = None
        for i in xrange(len(plans)):
            plan = plans[i]
            tensor = crops_tensor[i]
            if plan.flip:
                tensor = tf.image.flip_left_right(tensor)
            tensor = tf.reshape(tensor, (1, chip_r, chip_c, d))

            if flip_tensor is None:
                flip_tensor = tensor
            else:
                flip_tensor = tf.concat([flip_tensor, tensor], 0)

        return flip_tensor, boxes




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
            center = rect.center

            rand_translate = (np.random.uniform(-self._translate_amount, self._translate_amount) * rect.width,
                              np.random.uniform(-self._translate_amount, self._translate_amount) * rect.height)


            rand_scale = (np.random.uniform(*self._width_scale_range),
                          np.random.uniform(*self._height_scale_range))


            scale_size = (rect.width * rand_scale[0], rect.height * rand_scale[1])
            offset_center = Point2f(center[0] + rand_translate[0], center[1] + rand_translate[1])
            rand_rect = Rectanglef.create_with_center(offset_center, scale_size[0], scale_size[1])

        else:
            scale = np.random.uniform(0.1, 0.95)
            size = scale * np.min(r, c)
            point = Point2f(np.random.randint(0, 65535) % (c - size),
                      np.random.randint(0, 65535) % (r - size))

            rand_rect = Rectanglef.create_with_tlwh(point, size, size)

        return CropPlan(rand_rect, should_flip, angle)



    def gen_transform_mat(self, image_shape, anchor, translate=[0, 0], angle=0, scale=[1, 1]):
        anchor = np.array(anchor, dtype=np.float32)
        translate = np.array(translate, dtype=np.float32)
        scale = np.array(scale, dtype=np.float32)

        r, c, d = image_shape
        radian = np.deg2rad(angle)
        r_mat = np.array([[np.cos(radian), -np.sin(radian)],
                          [np.sin(radian), np.cos(radian)]], dtype=np.float32)

        r_mat[0, 0] /= scale[0]
        r_mat[1, 1] /= scale[1]

        t_vec = np.dot(r_mat, anchor * -1) + anchor + np.array(translate)
        mat = np.column_stack((r_mat, t_vec))
        mat = np.array([list(mat[0]) + list(mat[1]) + [0] * 2], dtype=np.float32)

        return mat




