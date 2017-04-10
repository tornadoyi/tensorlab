from __future__ import division

import tensorflow as tf

from input import Input
from model import Model
from loss import mmod_loss

from ..support import dataset
from ..support.image import RandomCrop
from ..support.utils import *



def load_data(file):
    def create_rectangle(t, l, w, h):
        return [t, l, t+h-1 , l+w-1]

    images, labels = dataset.load_object_detection_xml(file, create_rectangle)
    return images, labels



def main(datapath):
    crop_size = (200, 200)
    pyramid_scale = 6

    # create session
    sess = tf.InteractiveSession()

    # load train datas
    images, labels = load_data(datapath)


    # create crop generator
    croper = RandomCrop(
        images, labels,
        set_chip_dims = crop_size,
        probability_use_label = 0.5,
        max_roatation_angle = 30,
        translate_amount = 0.1,
        random_scale_range = (1.3, 4),
        probability_flip = 0.5,
        min_rect_ratio = 0.01,
        min_part_rect_ratio = 0.4)


    '''
    crop_images, crop_rect_list = croper(sess, 100)
    for i in xrange(len(crop_images)):
        image = crop_images[i].astype(np.uint8)
        rects = crop_rect_list[i]
        print(len(rects))
        for r in rects:
            y1, x1, y2, x2 = r
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.imshow("image", image)
        print("progress {0}/{1}".format(i+1, len(crop_images)))
        press_key_stop()
    exit()
    '''

    # create input layer
    input_layer = Input(sess, croper, crop_size, pyramid_scale)

    # create model
    is_training = tf.Variable(False, dtype=tf.bool)
    model = Model(input_layer, is_training)

    # create loss
    loss = mmod_loss(model, 40, 40)

    # init all variables
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_vars = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    # train
    while True:
        TAG_TIME()
        set_is_training = tf.assign(is_training, True)
        is_train = sess.run([set_is_training, is_training])[1]


        input_layer.debug_show(sess, 100)

        # debug pyramid image
        #input_layer.debug_show(sess, mini_batch_samples, mini_batch_labels)

        #result = model.run(sess, update_vars, feed_dict={input_layer.input: mini_batch_samples})
        #print("run model cost {0}".format(time_tag()))



    sess.close()


if __name__ == "__main__":
    main()

