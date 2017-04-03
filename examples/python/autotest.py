

import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.framework import *
from tensorlab.runtime.support import rectangle_yx as rt
import math
from support import dataset
import cv
from util import *
import threading
import time
from tensorflow.python.ops import control_flow_ops

images, labels = dataset.load_object_detection_xml("../data/testing.xml")

sess = tf.InteractiveSession()


def test_assign_image():
    image = np.array([images[0]])
    print image.shape
    _, h, w, c =  image.shape

    new_image = np.zeros((1, h*2, w*2, 3), dtype=image.dtype)
    new_image = tf.image.convert_image_dtype(new_image, dtype=tf.float32)
    result = tl.image.assign_image(image, new_image, [[0.25, 0.25, 0.75, 0.75]], [[0,0,0]])

    out_image = result.eval()[0]
    cv2.imshow("image", out_image.astype(np.uint8))
    press_key_stop()


def test_pyramid_plan():
    size, rects = tl.image.pyramid_plan((200, 150), 6)
    size, rects = sess.run([size, rects])
    print(size)

    image = np.zeros(size, dtype=np.uint8)
    rects = rt.convert_ratio_to_real(size, rects)
    for i in xrange(0, len(rects), 1):
        y1, x1, y2, x2 = rects[i]
        #print(y1, x1, y2, x2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    cv2.imshow("image", image)
    press_key_stop()


def test_pyramid_apply():
    image = np.array([images[0]] * 10)
    print image.shape
    b, h, w, c = image.shape

    (size, rects) = tl.image.pyramid_plan((h, w), 6)
    out_images = tl.image.pyramid_apply(image, size, rects)
    size, rects, out_images = sess.run([size, rects, out_images])

    out_images = out_images.astype(np.uint8)
    rects = rt.convert_ratio_to_real(size, rects)

    for i in xrange(b):
        img = out_images[i]
        for p in rects:
            y1, x1, y2, x2 = p
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.imshow("image", img)
        press_key_stop()




def test_pyramid_image():
    image = np.array([images[0]]*10)
    print image.shape
    b, h, w, c = image.shape

    out_images, rects = tl.image.pyramid(image, 6)
    out_images, rects = sess.run([out_images, rects])
    out_images = out_images.astype(np.uint8)

    _, height, width, _ = out_images.shape
    rects = rt.convert_ratio_to_real((height, width), rects)

    for i in xrange(b):
        image = out_images[i]
        for p in rects:
            y1, x1, y2, x2 = p
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.imshow("image", image)
        press_key_stop()




def test_flat_color():
    image = images[0]
    image = np.array([image, image, image, image, image])
    result = tl.image.flat_color(image)
    print(result)
    r = result.eval()
    r = r.astype(np.uint8)

    for i in xrange(r.shape[0]):
        print("{0} image".format(i))
        cv2.imshow("scale", r[i])
        press_key_stop()



def test_point_to_resize_space():
    points = np.random.rand(10, 2)
    scales = np.random.rand(15, 2)
    indexes = [ (i, i) for i in xrange(len(points))]

    map_points = tl.image.point_to_resize_space(points, scales, indexes)




if __name__ == '__main__':
    #test_assign_image()
    # test_pyramid_plan()
    # test_pyramid_apply()
    test_pyramid_image()
    # test_point_to_resize_space()
