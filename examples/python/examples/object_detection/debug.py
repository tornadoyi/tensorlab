import os
import pickle
import cv2
import numpy as np


def save_temp_data(crop_images, crop_rects, rect_groups):
    if not os.path.isdir("../temp_data"): os.mkdir("../temp_data")
    s_crop_images = pickle.dumps(crop_images)
    s_crop_rects = pickle.dumps(crop_rects)
    s_rect_groups = pickle.dumps(rect_groups)

    with open("../temp_data/image.txt", "w") as f: f.write(s_crop_images)
    with open("../temp_data/rect.txt", "w") as f: f.write(s_crop_rects)
    with open("../temp_data/group.txt", "w") as f: f.write(s_rect_groups)

    image = crop_images[0]
    for r in crop_rects:
        y1, x1, y2, x2 = r
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    cv2.imwrite("../temp_data/image.png", image)


def load_temp_data():
    with open("../temp_data/image.txt") as f: crop_images = pickle.loads(f.read())
    with open("../temp_data/rect.txt") as f: crop_rects = pickle.loads(f.read())
    with open("../temp_data/group.txt") as f: rect_groups = pickle.loads(f.read())
    return crop_images, crop_rects, rect_groups