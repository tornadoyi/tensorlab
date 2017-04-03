

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



x = tl.test.test(TArray([1,2,3]), TArray([4,5,6]))

print type(x)
print(x)

