
import os
import tensorflow as tf

module_path = os.path.dirname(__file__)
tl = tf.load_op_library(os.path.join(module_path, "libtensorlab.so"))

def pyramid(*args, **kwargs): return tl.pyramid(*args, **kwargs)

def assign_image(*args, **kwargs): return tl.assign_image(*args, **kwargs)

def flat_color(*args, **kwargs): return tl.flat_color(*args, **kwargs)
