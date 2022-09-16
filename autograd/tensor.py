#!/usr/bin/env python3

class Tensor:
    """ N-dimensional array which stores a scalar, vector
        or matrix of Values. """

    def __init__(self, data):
        if type(data) != np.ndarray:
            print("Error constructing tensor with %r" % data)
            assert(False)

        self.data = data
        self.grad = None
        
        self._ctx = None

