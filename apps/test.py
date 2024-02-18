import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import time
import os
import numpy as np
from needle.backend_ndarray import *
from needle.init import *

start = time.time()
a = ndl.Tensor(np.ones((1024, 1024)), device=triton())
b = ndl.Tensor(np.ones((1024, 1024)), device=triton())
c = a @ b
print(c)
end = time.time()
print(end - start)

start = time.time()
x = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
y = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
z = x @ y
print(z)
end = time.time()
print(end - start)