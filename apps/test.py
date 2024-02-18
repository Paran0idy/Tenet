import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import time
import os
from needle import backend_ndarray as nd


# a = nd.NDArray([1, 1], device=nd.triton())
# b = nd.NDArray([1, 1], device=nd.triton())



# c = a + b
# print(c, type(c))


x = ndl.Tensor([1, 2], device=nd.triton())
y = ndl.Tensor([1, 2], device=nd.triton())
print(x @ y, x.device)
