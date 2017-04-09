from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.framework import *
from tensorflow.contrib.framework import is_tensor
import math
from support import dataset
import cv
from util import *
import threading
import time
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import function
import cv2

images, labels = dataset.load_object_detection_xml("../data/testing.xml")

sess = tf.InteractiveSession()



array = tf.TensorArray(tf.int32, size=10, clear_after_read=False, infer_shape=False)

def body(s, array):
    v = tf.cond(tf.less(s.step, 1), lambda : tf.constant(0, shape=(1,1)), lambda : tf.constant(1, shape=(2,2)))
    return array.write(s.step, v)

#array = tl.for_loop(body, 0, 10, loop_vars=[array], shape_invariants=[tf.TensorShape(None)])


array = array.write(0, 0)
array = array.write(1, 1)

array.close()

v = array.gather([0])[0]

array = array.write(0, v+1)

v = array.read(0)

#array = array.write(3, v)

print(v.eval())

exit()

'''
for i in xrange(len(images)):
    image = images[i]
    print(image.shape)
    image = tf.convert_to_tensor(image)
    array = array.write(i, image)

'''

t1 = array.read(0)
t2 = array.read(1)

print(t1.eval())
print(t2.eval())

print(t1.shape)

#print(t.shape, " ", len(t.shape))

