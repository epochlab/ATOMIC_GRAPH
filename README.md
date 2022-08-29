# autograd

**Project ID:**  U8bTnBck

![alt text](https://github.com/epochlab/autograd/blob/main/sample.png)

#### Implementing backpropagation in a simple deep learning framwork.
Abstract: *Autograd is a deep learning framework written from scratch, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).*

`autograd.engine` module implements a scalar valued autograd engine, which tracks values, their gradients,
and the executed operations (and the resulting new values) in the form of a DAG (directed acyclical graph).

In this DAG the leaves are the input values and the roots are the outputs of the computational graph. 

By tracing this graph from root to leaves, you can calculate the gradient of each node by using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

This is called backpropagation, or more formally, [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation).

```python
from autograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Requirements
- Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
- 64-bit Python 3.7.9 installation.