#!/usr/bin/env python3

from re import S
import numpy as np

class Tensor:
    """ N-dimensional array which stores a scalar, vector
        or matrix of Values. """

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None

    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def shape(self):
        return self.data.shape