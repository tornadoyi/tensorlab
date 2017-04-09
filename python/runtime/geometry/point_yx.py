import numpy as np

def __call__(y, x, dtype=None): return np.array((y, x), dtype)

def create(y, x, dtype=None): return np.array((y, x), dtype)

def y(p): return p[0]

def x(p): return p[1]

