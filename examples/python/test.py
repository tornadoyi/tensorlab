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

x = tf.placeholder(tf.int32, [None, None], name="x")



a = tf.constant([1,2,3,4,5], tf.float32)

b = a + tf.sparse_to_dense([2], tf.shape(a), 0.1)

#c = tf.sparse_tensor_to_dense(c)

r = sess.run(b)

print(r)
