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
from support import dataset






images, labels = dataset.load_object_detection_xml("../data/testing.xml")

sess = tf.InteractiveSession()



array = tf.TensorArray(tf.int32, size=10, clear_after_read=False, infer_shape=False)

def body(s, array):
    v = tf.cond(tf.less(s.step, 1), lambda : tf.constant(0, shape=(1,1)), lambda : tf.constant(1, shape=(2,2)))
    return array.write(s.step, v)

#array = tl.for_loop(body, 0, 10, loop_vars=[array], shape_invariants=[tf.TensorShape(None)])


array = array.write(0, 0)
array = array.write(1, 1)



