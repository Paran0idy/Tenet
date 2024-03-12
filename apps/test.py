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
c = a @ b
