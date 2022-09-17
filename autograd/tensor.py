#!/usr/bin/env python3

from re import S
import numpy as np

class Tensor:
    """ N-dimensional array which stores a scalar [n0], vector [n1]
        or matrix [n2] of Tensors. """

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None

    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)

    def __add__(self, other): return Tensor(self.data + other.data)
    def __mul__(self, other): return Tensor(self.data * other.data)
    def __sub__(self, other): return Tensor(self.data + (-other.data))

    def shape(self): return self.data.shape

    # def backward(self):
    #     print(self.data.shape)
    #     self.grad = np.ones(*self.data.shape)
        