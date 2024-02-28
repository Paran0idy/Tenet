import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import os
import numpy as np
from needle.backend_ndarray import *

print("----Triton Backend----")

a = ndl.Tensor(np.ones((1024, 1024)), device=triton())
b = ndl.Tensor(np.ones((1024, 1024)), device=triton())
c = a - b
c.backward()

print("c : \n", c)
print("a.grad : \n", a.grad)
print("b.grad : \n", b.grad)


print("----CPU Backend----")

x = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
y = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
z = x @ y
z.backward()

print("z : \n", z)
print("x.grad : \n", x.grad)
