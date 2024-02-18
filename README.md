# Tenet
A DL Framework for Tensor Computation

Inspired by Needle Framework in CMU 10-414/714: Deep Learning Systems
https://dlsyscourse.org/


## Example
```python
import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
from needle.backend_ndarray import *

# Triton
a = ndl.Tensor(np.ones((1024, 1024)), device=triton())
b = ndl.Tensor(np.ones((1024, 1024)), device=triton())
c = a @ b
print(c)

# CPU
x = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
y = ndl.Tensor(np.ones((1024, 1024)), device=cpu())
z = x @ y
print(z)
```


```
[[1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 ...
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]]

[[1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 ...
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]
 [1024. 1024. 1024. ... 1024. 1024. 1024.]]
 ```

## Autograd

## Module
+ Module
+ Optim

## Backend
+ OpenAI Triton: MMA instruction using Tensor Cores
+ NVIDIA CUDA
+ X86 CPU

