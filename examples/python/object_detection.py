import math
import cv
import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.ops import batch_normalization as bn
from util import *
from support import dataset
from support.image import RandomCrop
import time

def load_data(file):
    def create_rectangle(t, l, w, h):
        p = tl.Point2f(t, l)
        return tl.Rectanglef.create_with_tlwh(p, w, h)

    images, labels = dataset.load_object_detection_xml(file, create_rectangle)
    return images, labels



def create_model(input):
    def weight(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def conv2d(x, W, strides, padding):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding)

    tensor = input
    tensor = conv2d(tensor, weight([5, 5, 3, 32]), [1, 2, 2, 1], padding="VALID")
    tensor = conv2d(tensor, weight([5, 5, 32, 32]), [1, 2, 2, 1], padding="VALID")
    tensor = conv2d(tensor, weight([5, 5, 32, 32]), [1, 2, 2, 1], padding="VALID")

    tensor = conv2d(tensor, weight([3, 3, 32, 32]), [1, 1, 1, 1], padding="SAME")
    tensor = conv2d(tensor, weight([3, 3, 32, 32]), [1, 1, 1, 1], padding="SAME")
    tensor = conv2d(tensor, weight([3, 3, 32, 32]), [1, 1, 1, 1], padding="SAME")

    tensor = conv2d(tensor, weight([6, 6, 32, 1]), [1, 1, 1, 1], padding="SAME")

    return tensor



def debug_images_rects(sess, images_tensor, rects_list):
    result = sess.run([images_tensor])
    images = result[0]
    for i in xrange(len(images)):
        img = images[i]
        rects = rects_list[i]
        for r in rects:
            cv2.rectangle(img, (int(r.left), int(r.top)), (int(r.right), int(r.bottom)), color=(0, 0, 255),
                          thickness=2)

        cv2.imshow("image", img)
        print("progress {0}/{1} rect:{2}".format(i + 1, len(images), len(rects)))
        press_key_stop()


def main():
    crop_size = (200, 200)
    crop_per_image = 30
    pyramid_scale = 6

    # load train datas
    images, labels = load_data("../data/training.xml")

    # create crop generator
    croper = RandomCrop(crop_size)

    # create model
    input = tf.placeholder(tf.float32, (None, None, None, None))
    model = create_model(input)


    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    while True:
        mini_batch_samples, mini_batch_labels = croper(images, labels, crop_per_image)
        mini_batch_samples = tl.image.pyramid(mini_batch_samples, pyramid_scale)
        debug_images_rects(sess, mini_batch_samples, mini_batch_labels)
        exit()

    sess.close()


if __name__ == "__main__":
    main()

