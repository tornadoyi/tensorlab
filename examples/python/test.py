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

v = tf.Variable(0.0)
v1 = tf.Variable(0.0)
op_v = v.assign_add(1.0)
loss = (1.0 - v * 3.0) ** 2.0

loss1 = (2.0 - v1 * 4.0) ** 2.0

with tf.variable_scope("optimizer"):
    optimizer = tf.train.RMSPropOptimizer(0.1)

with tf.variable_scope("train_step"):
    train_step = optimizer.minimize(loss)

with tf.variable_scope("train_step2"):
    train_step2 = optimizer.minimize(loss1)

slot_names = optimizer.get_slot_names()

vs = []
for name in slot_names:
    vs.append(optimizer.get_slot(v, name))

vs1 = []
for name in slot_names:
    vs1.append(optimizer.get_slot(v1, name))

for p in vs: print(p)
print("*"*100)
for p in vs1: print(p)

train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) #+ tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)

print(vs[0] == vs1[0])
print(vs[1] == vs1[1])

sess.run(tf.global_variables_initializer())






saver = tf.train.Saver(train_vars)


trainer = Trainer(sess, saver, checkpoint="checkpoints/test/test.ckpt", max_epoch=5, max_save_epoch=1, save_with_epoch=True)

for v in train_vars:
    print(v.name, v.eval())

print("="*100)

def step_call_back(r):
    for v in train_vars:
        print(v.name, v.eval())

    print("="*100)


trainer(train_step, epoch_callback=step_call_back)


for v in train_vars:
    print(v.name, v.eval())

print("="*100)