

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

images, labels = dataset.load_object_detection_xml("../data/testing.xml")

images = np.array([images[0], images[0], images[0], images[0], images[0]])

image = np.array([images[0]])
simage = images[0]

a = np.array([[1,2,3], [4,5,6], [7,8,9]])


def gen_transform_mat(image_shape, anchor, translate=[0, 0], angle=0, scale=[1, 1]):
    anchor = np.array(anchor, dtype=np.float32)
    translate = np.array(translate, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)

    r, c, d = image_shape
    radian = np.deg2rad(angle)
    r_mat = np.array([[np.cos(radian), np.sin(radian)],
                      [-np.sin(radian), np.cos(radian)]], dtype=np.float32)

    r_mat[0, 0] /= scale[0]
    r_mat[1, 1] /= scale[1]

    t_vec = np.dot(r_mat, anchor * -1) + anchor + np.array(translate)
    mat = np.column_stack((r_mat, t_vec))
    mat = np.array([list(mat[0]) + list(mat[1]) + [0] * 2], dtype=np.float32)

    return mat

b, r, c, d = image.shape
mat = gen_transform_mat((r, c, d), (c/2, r/2), angle=30)
tensor = tf.contrib.image.transform(image, mat)

with tf.Session() as sess:
    new_image = tensor.eval()[0]
    cv2.imshow("crop", new_image)


mat = mat[0]
r_mat = np.array([mat[[0,1]], mat[[3,4]]])
t_vec = mat[[2,5]]

new_image = np.zeros_like(simage)

for i in xrange(r):
    for j in xrange(c):
        point = np.array([j, i])
        new_point = np.dot(r_mat, point) + t_vec
        if new_point[0] < 0 or new_point[0] > c: continue
        if new_point[1] < 0 or new_point[1] > r: continue
        nx, ny = new_point[0], new_point[1]
        new_image[ny, nx] = simage[i, j]

cv2.imshow("mine", new_image)



rot_mat = cv2.getRotationMatrix2D((c/2, r/2), -30, 1)
rot_mat = np.column_stack((r_mat, t_vec))
new_image = cv2.warpAffine(simage, rot_mat, (c, r), flags=cv2.INTER_LANCZOS4)
cv2.imshow("cv", new_image)


press_key_stop()

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