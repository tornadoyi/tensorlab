

import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.framework import *
import math
import dataset
import cv
from util import *
import threading
import time

images, labels = dataset.load_xml("../data/testing.xml")

images = np.array([images[0], images[0], images[0], images[0], images[0]])

image = np.array([images[0]])

'''
exit_flag = False

# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.
def MyLoop(coord, index):
    starttime = time.time()
    last_time = time.time()
    while not coord.should_stop():
        curtime = time.time()
        if curtime - last_time > 3:
            print("thread {0} works".format(index))
            last_time = curtime

        for i in xrange(1000000):
            t = index / 3

        if curtime - starttime > 30:
            print("request stop")
            coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in xrange(100)]

# Start the threads and wait for all of them to stop.
for t in threads:
  t.start()
coord.join(threads)



exit()
'''

'''
model = Model()
x1 = model.add_input(tf.int32, [None, None], name="x1")
x2 = model.add_input(tf.int32, [None, None])
res = Layer(tf.add)

x1.adds([
    Layer(tf.multiply, y=2),
    Layer(tf.add, y=3),
])

x2.adds([
    Layer(tf.div, y=2),
    Layer(tf.add, y=-7),
])


res.adds([
    Layer(tf.cast, dtype=tf.float32),
    Layer(tf.log),
])

x1.tail.add(res)
x2.tail.add(res)

model.summary()

with tf.Session() as sess:
    x = np.random.randint(1, 10, size=(3, 3))
    y = np.random.randint(1, 10, size=(3, 3))

    x = np.array([[1, 1, 1], [2, 2, 2]])
    y = np.array([[3, 3, 3], [4, 4, 4]])

    result = model.run(sess, feed_dict = {"x1": x, x2: y})

    for k, v in result.items():
        print(k)
        print(v)
        print('-' * 10)

'''