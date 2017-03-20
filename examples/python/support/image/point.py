import numpy as np


class Point(np.array):
    def __init__(self, *args):
        np.array.__init__(self, args)


    @property
    def x(self): return self[0]

    @x.setter
    def x(self, v): self[0] = v

    @property
    def y(self): return self[1]

    @y.setter
    def y(self, v): self[1] = v

    @property
    def z(self): return self[2]

    @z.setter
    def z(self, v): self[2] = v