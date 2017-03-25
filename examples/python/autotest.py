import sys
import select
import numpy as np
import tensorflow as tf
import tensorlab as tl
import cv2
from time import sleep

images, labels = dataset.load_xml("../data/testing.xml")
#image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)


def press_key_stop():
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
            break
        else:
            cv2.waitKey(10)


def test_assign_image():
    image = images[0]
    image = np.array([image, image, image, image, image])
    shape = image.shape
    out_image = np.zeros((shape[0], shape[1]*2,  shape[2]*2, shape[3]))
    result = tl.image.assign_image(image, out_image, [0, 0])
    result = tl.image.assign_image(image, result, (shape[1], shape[2]))
    r = result.eval()

    for i in xrange(r.shape[0]):
        print("{0} image".format(i))
        cv2.imshow("scale", r[i])
        press_key_stop()


def test_pyramid():
    image = images[0]
    image = np.array([image, image, image, image, image])
    result = tl.image.pyramid(image, 6)
    r = result.eval()
    r = r.astype(np.uint8)

    for i in xrange(r.shape[0]):
        print("{0} image".format(i))
        cv2.imshow("scale", r[i])
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


with tf.Session():
    test_flat_color()


