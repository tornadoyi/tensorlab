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



def body():
    a = tf.constant(0)
    b = tf.constant(1)

    a = tl.Print(a, message="a ")
    b = tl.Print(b, message="b ")

    return tf.cond(tf.constant(True), lambda :a, lambda: b)

r = tf.cond(tf.constant(False), lambda : tf.constant(-1), lambda : body())

r = sess.run(r)

print(r)
