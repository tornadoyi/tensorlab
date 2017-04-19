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

gloabl_i = 1
def gen_samples():
    global gloabl_i
    w = np.array([4,3,2,1], np.float32)
    x = np.array([1,1,1,1]) * gloabl_i#np.random.randint(0, 10, size=4)
    y = w * x
    return x, y

x = tf.placeholder(tf.int32, [None, ])
y = tf.placeholder(tf.int32, [None, ])

w = tf.Variable([1,2,3,4], dtype=tf.float32, name="vv")
y_ = w * tf.to_float(x)

loss = tf.nn.l2_loss(tf.to_float(y) - y_)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

def gen_feed_dict():
    sx, sy = gen_samples()
    return {x: sx, y:sy}

trainer = Trainer(sess, checkpoint="checkpoint/test/test.ckpt", max_epoch=10, max_save_epoch=5)


def step_call_back(r):
    print(trainer.epoch)
    print("loss", r[1])


trainer([train_step, loss], gen_feed_dict, step_call_back)