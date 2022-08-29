# AUTOGRAD

**Project ID:**  U8bTnBck

![alt text](https://github.com/epochlab/autograd/blob/main/sample.png)

--------------------------------------------------------------------

#### A simple deep learning framework.
Abstract: *A scalar-valued autograd engine which tracks values, gradients and executed operations over a dynamically built DAG (directed acyclical graph).*

### Example

```python
from autograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).tanh()
d += 3 * d + (b - a).tanh()
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

### Acknowledgments
[micrograd](https://github.com/karpathy/micrograd) (2020)<br />
[tinygrad](https://github.com/geohot/tinygrad) (2022)
