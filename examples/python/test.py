

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
with tf.Session() as sess:
    b, r, c, d = image.shape
    image = tf.image.convert_image_dtype(image, tf.float32)
    new_image = tf.Variable(np.zeros((b, r*2, c*2, d)), dtype=tf.float32)
    new_image.assign(image)
    new_image = new_image.eval()
    cv2.imshow("origin", new_image[0])

exit()

'''
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

image = images[0]
new_image = rotate_about_center(image, 30)
print(np.shape(new_image))
cv2.imshow("origin", new_image)
press_key_stop()
'''


def gen_transform_mat(image, anchor, translate=(0, 0), angle=0, scale = (1, 1)):
    anchor = np.array(anchor, dtype=np.float32)
    translate = np.array(translate, dtype=np.float32)
    scale = np.array(scale, dtype=np.float32)

    r, c, d = np.shape(image)
    radian = np.deg2rad(angle)
    r_mat = np.array([[np.cos(radian), -np.sin(radian)],
                      [np.sin(radian), np.cos(radian)]], dtype=np.float32)


    r_mat[0, 0] *= scale[0]
    r_mat[1, 1] *= scale[1]

    t = np.dot(r_mat, anchor * -1) + anchor + np.array(translate)
    mat = np.column_stack((r_mat, t))

    return mat


image = images[0]
r, c, d = np.shape(image)
mat = gen_transform_mat(image, (c/2, r/2), translate=(20,20))

#new_image = cv2.warpAffine(image, mat, (c, r), flags=cv2.INTER_LANCZOS4)
with tf.Session() as sess:
    mat = np.array([list(mat[0]) + list(mat[1]) + [0]*2], dtype=np.float32)
    print(mat.shape); print(mat)
    image = np.array([image])
    print(image.shape)
    image = tf.image.convert_image_dtype(image, tf.float32)
    new_image = tf.contrib.image.transform(image, mat)
    new_image = new_image.eval()
    print(new_image[0][100:150])
cv2.imshow("origin", new_image[0])
press_key_stop()
exit()

boxes = []
for img in images:
    r, c = 1,1
    y1 = np.random.uniform(r/4, r/2)
    x1 = np.random.uniform(c/4, c/2)
    y2 = np.random.uniform(r/2, r)
    x2 = np.random.uniform(c/2, c)
    box = [y1, x1, y2, x2]
    boxes.append(box)


#boxes = np.array(boxes)
box_ind = np.array(range(len(images)), dtype=np.int32)

print(images.dtype, np.shape(images))
#print(boxes.dtype, np.shape(boxes))
print(box_ind.dtype, np.shape(box_ind))

print(boxes)
print(box_ind)

with tf.Session() as sess:

    images = tf.image.convert_image_dtype(images, tf.float32)
    t = tf.image.crop_and_resize(images, boxes=boxes, box_ind=box_ind, crop_size=(200, 200))
    res = t.eval()

    for img in res:
        print(img[30:50])
        cv2.imshow("window", img)
        press_key_stop()

#tf.image.crop_and_resize(image, )

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