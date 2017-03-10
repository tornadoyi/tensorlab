
import os
import inspect
import tensorflow as tf

module_path = os.path.dirname(__file__)
tl = tf.load_op_library(os.path.join(module_path, "libtensorlab.so"))

def_function = "def {0}(*args, **kwargs): return tl.{0}(*args, **kwargs)"

for name in dir(tl):
    f = getattr(tl, name)
    if not inspect.isfunction(f): continue
    exec(def_function.format(name))

