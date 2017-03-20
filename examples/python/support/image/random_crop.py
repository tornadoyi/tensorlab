import os
import numpy as np
from collections import namedtuple
from tensorlab import Rectanglef, Point2f

CropPlan = namedtuple("CropPlan", ["rect", "flip", "angle"])


class RandomCrop(object):
    def __init__(self,
                 set_chip_dims,
                 width_scale_range=(0.8, 1.3),
                 height_scale_range = (0.8, 1.3),
                 max_roatation_angle = 30,
                 probability_use_label = 0.5,
                 translate_amount = 0.1,
                 probability_flip = 0.5,):
        self._chips_dims = set_chip_dims
        self._width_scale_range = width_scale_range
        self._height_scale_range = height_scale_range
        self._max_roatation_angle = max_roatation_angle
        self._probability_use_label = probability_use_label
        self._translate_amount = translate_amount



    def __call__(self, images, labels):
        pass




    def _make_plan(self, image, rects):
        r, c, d = np.shape(image)
        should_flip = np.random.uniform(0, 1) > self.probability_flip
        angle = np.random.uniform(-self._max_roatation_angle, self._max_roatation_angle)

        # get rect from labels
        plan = None
        if np.random.uniform(0, 1) < self._probability_use_label:
            index = np.random.randint(0, len(rects))
            rect = np.array(rects[index])
            center = rect.center

            rand_translate = (np.random.uniform(-self._translate_amount, self._translate_amount) * rect.width,
                              np.random.uniform(-self._translate_amount, self._translate_amount) * rect.height)


            rand_scale = (np.random.uniform(*self._width_scale_range),
                          np.random.uniform(*self._height_scale_range))


            scale_size = (rect.width * rand_scale[0], rect.height * rand_scale[1])
            offset_center = Point2f(center[0] + rand_translate[0], center[1] + rand_translate[1])
            res_rect = Rectanglef.create_with_center(offset_center, scale_size[0], scale_size[1])

            plan = CropPlan(res_rect, should_flip, angle)

        else:
            scale = np.random.uniform(0.1, 0.95)
            size = scale * np.min(r, c)
            point = Point2f(np.random.randint(0, 65535) % (c - size),
                      np.random.randint(0, 65535) % (r - size))

            return Rectanglef.create_with_tlwh(point, size, size)