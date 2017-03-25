

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

tcond = tf.Variable(True, dtype=tf.bool)

c1 = tf.constant(1)
c2 = tf.constant(0)

cond = control_flow_ops.cond(tcond, lambda : c1, lambda : c2)

with tf.Session() as sess:
    print(c1.name)
    print(c2.name)


