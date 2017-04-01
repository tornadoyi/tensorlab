

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

sess = tf.InteractiveSession()


def test_pyramid_apply():
    image = np.array([images[0]] * 10)
    print image.shape
    b, h, w, c = image.shape

    plan = tl.image.pyramid_plan((h, w), 6)
    out_images = tl.image.pyramid_apply(image, plan)
    out_images = out_images.eval()
    out_images = out_images.astype(np.uint8)

    plan = plan.eval()[1:]

    for i in xrange(b):
        img = out_images[i]
        for p in plan:
            y, x, h, w = p;
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color=(255, 0, 0), thickness=2)
        cv2.imshow("image", img)
        press_key_stop()

test_pyramid_apply()


def test_pyramid_image():
    image = np.array([images[0]]*10)
    print image.shape
    b, h, w, c = image.shape

    result = tl.image.pyramid(image, 6)
    out_images = result.eval()
    out_images = out_images.astype(np.uint8)
    for i in xrange(b):
        cv2.imshow("image", out_images[i])
        press_key_stop()

#test_pyramid_image()


def test_assign_image():
    image = np.array([images[0]])
    print image.shape
    _, h, w, c =  image.shape

    new_image = np.zeros((1, h*2, w*2, 3), dtype=image.dtype)
    new_image = tf.image.convert_image_dtype(new_image, dtype=tf.float32)
    result = tl.image.assign_image(image, new_image, [[h/2, w/2, h, w]], [[0,0,0]])

    out_image = result.eval()[0]
    cv2.imshow("image", out_image.astype(np.uint8))
    press_key_stop()

#test_assign_image()




def test_pyramid_plan():
    results = tl.image.pyramid_plan((200, 150), 6)

    rects = results.eval()
    _, _, height, width = rects[0]
    print(height)
    print(width)

    image = np.zeros((height, width), dtype=np.uint8)

    for i in xrange(1, len(rects), 1):
        y, x, h, w = rects[i]
        print(y, x, h, w)
        cv2.rectangle(image, (x, y), (x + w-1, y + h-1), color=(255, 0, 0), thickness=2)

    cv2.imshow("image", image)
    press_key_stop()


