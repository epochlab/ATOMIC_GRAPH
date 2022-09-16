#!/usr/bin/env python3

from re import S
import numpy as np

class Tensor:
    """ N-dimensional array which stores a scalar, vector
        or matrix of Values. """

    def __init__(self, data, _children=(), _op=''):
        if type(data) != np.ndarray:
            print("Error constructing tensor with %r" % data)
            assert(False)

        self.data = np.array(data, dtype=np.float32)
        self.shape = data.shape
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Bad mutual shapes, inner dimensions must match.")
        return Tensor(self.data + other.data, (self, other), _op='+')

    def __mul__(self, other):
        if self.shape != other.shape:
            raise ValueError("Bad mutual shapes, inner dimensions must match.")
        return Tensor(self.data * other.data, (self, other), _op='*')

    