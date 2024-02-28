import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import time
import os
import numpy as np
from needle.backend_ndarray import *

start = time.time()
a = ndl.Tensor(np.ones((1024, 1024)), device=triton())
b = ndl.Tensor(np.ones((1024, 1024)), device=triton())
c = a - b
c.backward()

print(c, a.grad)
end = time.time()
print(end - start)

start = time.time()
x = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
y = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
z = x @ y
z.backward()
print(x.grad, x.device)

print(z)
end = time.time()
print(end - start)