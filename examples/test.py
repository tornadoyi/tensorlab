import sys
import select
import numpy as np
import tensorflow as tf
import tensorlab as tl
import dataset
import cv2
from time import sleep

images, labels = dataset.load_xml("data/testing.xml")
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
    shape = image.shape
    concat_image = np.zeros((shape[0] * 2, shape[1], 3))
    result = tl.assign_image(image, concat_image, [0, 0])
    result = tl.assign_image(image, result, (shape[0], 0))
    return result.eval()


with tf.Session():
    r = test_assign_image()
    #r = r.astype(np.uint8)

    cv2.imshow("scale", r)
    cv2.imwrite("test.png", r)

    press_key_stop()