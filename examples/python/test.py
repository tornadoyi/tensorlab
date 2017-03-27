

import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.framework import *
import math
from support import dataset
import cv
from util import *
import threading
import time
from tensorflow.python.ops import control_flow_ops

images, labels = dataset.load_object_detection_xml("../data/testing.xml")



model = Model()

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

model.add(tf.add, a, b)
model.add(tf.add, model.out, 4)
model.add(tf.add, model.out, 1)

with tf.Session() as sess:
    result = model.run(sess, feed_dict={a : 1, b: 2})

    print(result[model.out])