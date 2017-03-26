

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

inputs = tf.placeholder(tf.float32, (3,3))
y = tf.convert_to_tensor(np.zeros((3,3)), dtype=tf.float32)

is_training = tf.Variable(True, dtype=tf.bool, trainable=False)

bn = tf.layers.batch_normalization(inputs, training=is_training)
bn = tf.layers.batch_normalization(bn, training=is_training)
loss = tf.reduce_sum((y - bn) **2 / 2)
train_step = loss#tf.train.AdamOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print(len(train_ops), len(update_ops))
    print(train_ops)

    for i in xrange(10):
        x = np.array([[i, i+1, i+2], [i, i-1, i-2], [i, i*1, i*2]], dtype=np.float)
        res = sess.run([train_step, loss] + update_ops, feed_dict={inputs: x})
        #print(len(train_ops))
        #print(res[2:4])
        #print(len(update_ops))
        #print(res[4:6])
        print(res[1])

        print("="*100)
        print("")

    set_is_training = tf.assign(is_training, False)
    is_train = sess.run([set_is_training, is_training])[1]
    print(is_train)

    x = np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float)
    res = sess.run([train_step, loss], feed_dict={inputs: x})
    print(res[1])