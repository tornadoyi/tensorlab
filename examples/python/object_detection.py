import math
import cv
import tensorflow as tf
import numpy as np
import tensorlab as tl
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



def create_model(input, is_training):
    def weight(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W, b, strides, padding):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding) + b

    def bn(inputs, is_traning):
        return tf.layers.batch_normalization(inputs, training=is_traning)

    tensor = input
    tensor = conv2d(tensor, weight([5, 5, 3, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
    tensor = bn(tensor, is_training)
    tensor = tf.nn.relu(tensor)

    tensor = conv2d(tensor, weight([5, 5, 32, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
    tensor = bn(tensor, is_training)
    tensor = tf.nn.relu(tensor)

    tensor = conv2d(tensor, weight([5, 5, 32, 32]), bias([32]), [1, 2, 2, 1], padding="VALID")
    tensor = bn(tensor, is_training)
    tensor = tf.nn.relu(tensor)

    tensor = conv2d(tensor, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
    tensor = bn(tensor, is_training)
    tensor = tf.nn.relu(tensor)

    tensor = conv2d(tensor, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
    tensor = bn(tensor, is_training)
    tensor = tf.nn.relu(tensor)

    tensor = conv2d(tensor, weight([3, 3, 32, 32]), bias([32]), [1, 1, 1, 1], padding="SAME")
    tensor = bn(tensor, is_training)
    tensor = tf.nn.relu(tensor)

    tensor = conv2d(tensor, weight([6, 6, 32, 1]), bias([1]), [1, 1, 1, 1], padding="SAME")

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
    is_training = tf.Variable(False, dtype=tf.bool)
    model = create_model(input, is_training)


    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    while True:
        set_is_training = tf.assign(is_training, True)
        is_train = sess.run([set_is_training, is_training])[1]
        print(is_training)

        mini_batch_samples, mini_batch_labels = croper(images, labels, crop_per_image)
        mini_batch_samples = tl.image.pyramid(mini_batch_samples, pyramid_scale)
        #debug_images_rects(sess, mini_batch_samples, mini_batch_labels)

        mini_batch_samples = sess.run([mini_batch_samples])[0]
        sess.run([model]+update_vars, feed_dict={input: mini_batch_samples})
        exit()


    sess.close()


if __name__ == "__main__":
    main()

