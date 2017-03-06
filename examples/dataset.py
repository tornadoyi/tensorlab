import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import cv2

def load_xml(config_path):
    dom = xml.dom.minidom.parse(config_path)
    root = dom.documentElement
    images = root.getElementsByTagName('images')[0]
    all_image = images.getElementsByTagName('image')

    data_path = os.path.dirname(config_path)
    image_list = []
    label_list = []
    for image in all_image:
        filename = image.getAttribute("file")
        boxes = image.getElementsByTagName("box")
        label_box = []
        for box in boxes:
            top = int(box.getAttribute("top"))
            left = int(box.getAttribute("left"))
            width = int(box.getAttribute("width"))
            height = int(box.getAttribute("height"))

            label_box.append((top, left, width, height))

        image_path = os.path.join(data_path, filename)

        img = load_image(image_path)

        image_list.append(img)
        label_list.append(label_box)


    return image_list, label_list


def load_image(filename):

    def PIL_load(filename):
        img = Image.open(filename)
        width, height = img.size
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in xrange(0, height):
            for x in xrange(0, width):
                p = img.getpixel((x, y))
                img_array[y, x][0:] = p
        img.close()
        return img_array

    def CV_load(filename):
        return cv2.imread(filename)

    return CV_load(filename)