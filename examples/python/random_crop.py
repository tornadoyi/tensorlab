import os
import numpy as np
from collections import namedtuple

CropPlan = namedtuple("CropPlan", ["flip", "angle", "rect"])


class RandomCrop(object):
    def __init__(self,
                 set_chip_dims,
                 width_scale_range=(0.8, 1.3),
                 height_scale_range = (0.8, 1.3),
                 max_roatation_angle = 30,
                 probability_use_label = 0.5,
                 translate_amount = 0.1):
        self._chips_dims = set_chip_dims
        self._width_scale_range = width_scale_range
        self._height_scale_range = height_scale_range
        self._max_roatation_angle = max_roatation_angle
        self._probability_use_label = probability_use_label
        self._translate_amount = translate_amount



    def __call__(self, *args, **kwargs):
        pass




    def _make_plan(self, image, labels):
        flip = np.random.uniform(0, 1) > 0.5
        angle = np.random.uniform(-self._max_roatation_angle, self._max_roatation_angle)

        # get rect from labels
        if np.random.uniform(0, 1) < self._probability_use_label:
            index = np.random.randint(0, len(labels))
            (top, left, width, height) = labels[index]

            rand_translate = (np.random.uniform(-self._translate_amount, self._translate_amount) * width,
                              np.random.uniform(-self._translate_amount, self._translate_amount) * height)


            rand_scale = (np.random.uniform(*self._width_scale_range),
                          np.random.uniform(*self._height_scale_range),)

