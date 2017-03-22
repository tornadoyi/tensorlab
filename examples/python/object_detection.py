import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.framework import *
import math
import cv
from util import *
from support import dataset
from support.image import RandomCrop



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


def create_rectangle(t, l, w, h):
    p = tl.Point2f(t, l)
    return tl.Rectanglef.create_with_tlwh(p, w, h)
images, labels = dataset.load_object_detection_xml("../data/testing.xml", create_rectangle)

rand_crop = RandomCrop((200, 200))
images_tensor, boxes = rand_crop(images, labels, 100)

with tf.Session() as sess:
    crop_images = images_tensor.eval()
    print("crop_images.shape", crop_images.shape)

    for img in crop_images:
        cv2.imshow("crop", img)
        press_key_stop()

