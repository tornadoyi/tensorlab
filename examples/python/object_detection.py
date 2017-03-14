

import tensorflow as tf
import numpy as np
import tensorlab as tl
from tensorlab.framework import *


model = Model()
x1 = model.add_input(tf.int32, [None, None], name="x1")
x2 = model.add_input(tf.int32, [None, None])
res = Layer(tf.add)

x1.adds([
    Layer(tf.multiply, y=2),
    Layer(tf.add, y=3),
])

x2.adds([
    Layer(tf.div, y=2),
    Layer(tf.add, y=-7),
])


res.adds([
    Layer(tf.cast, dtype=tf.float32),
    Layer(tf.log),
])

x1.tail.add(res)
x2.tail.add(res)

model.summary()

with tf.Session() as sess:
    x = np.random.randint(1, 10, size=(3, 3))
    y = np.random.randint(1, 10, size=(3, 3))

    x = np.array([[1, 1, 1], [2, 2, 2]])
    y = np.array([[3, 3, 3], [4, 4, 4]])

    result = model.run(sess, feed_dict = {"x1": x, x2: y})

    for k, v in result.items():
        print(k)
        print(v)
        print('-' * 10)