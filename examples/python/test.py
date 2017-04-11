from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv
import cv2
import threading
import time
import math
import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab.framework import *
from tensorflow.contrib.framework import is_tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import function
from examples.support import dataset
from tensorlab.ops.geometry import rectangle_yx as rt, point_yx as pt





images, labels = dataset.load_object_detection_xml("../data/testing.xml")

sess = tf.InteractiveSession()



rects = np.array([
    [0, 0, 9, 9],
    [10, 0, 19, 9],
    [20, 0, 29, 9],
    [30, 0, 39, 9],
    [40, 0, 49, 9],
])


points = np.array([
    [5, 5],
    [15, 5],
    [25, 5],
    [35, 5],
    [45, 5],
    [10, 10],
    [20, 11],
    [30, 12],
    [40, 13],
    [50, 14],
])


rects = tf.cast(rects, tf.int32)
points = tf.cast(points, tf.int32)

rect_count = tf.shape(rects)[0]
point_count = tf.shape(points)[0]

ymin, ymax = rt.top(rects), rt.bottom(rects)
xmin, xmax = rt.left(rects), rt.right(rects)

min = tf.concat([tf.expand_dims(ymin, 1), tf.expand_dims(xmin, 1)], axis=1)
max = tf.concat([tf.expand_dims(ymax, 1), tf.expand_dims(xmax, 1)], axis=1)


points = tf.tile(points, [1, rect_count])
points = tf.reshape(points, [-1, 2])

min = tf.tile(min, (point_count, 1))
max = tf.tile(max, (point_count, 1))


nearest_points = tf.clip_by_value(points, min, max)

vec = nearest_points - points


length_square = pt.length_square(vec)
length_square = tf.reshape(length_square, (point_count, -1))


indexes = tf.argmin(length_square, 1)


x_count = 100
a_count = 10
b_count = 10


for x in xrange(0, x_count):
    for a in np.arange(1.0, a_count, 0.1):
        for b in np.arange(0, b_count, 0.1):
            y = int(a * float(x) + b + 0.5)
            x1 = int((float(y) - b) / a + 0.5)

            #if x == 4 and a == 1.0 and b

            if x1 != x :
                print("x:{0} != x1:{4} a:{1} b:{2} y:{3} sub:{5}".format(x, a, b, y, x1, x1-x))






