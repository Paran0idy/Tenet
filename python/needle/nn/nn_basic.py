"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import needle as ndl

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variables
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X = X @ self.weight
        if self.bias:
            return X + self.bias.broadcast_to(X.shape)
        return X


class Flatten(Module):
    def forward(self, X):
        return X.reshape((X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * (x.cached_data > 0)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = ndl.Tensor(init.one_hot(logits.shape[1], y))
        y_new = ndl.summation(y_one_hot * logits, axes=(1, ))
        return  (ndl.logsumexp(logits, axes=(1, )).sum() - y_new.sum()) / logits.shape[0]

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        # Parameters
        self.weight = Parameter(Tensor(np.ones((dim))), device=device, dtype=dtype)
        self.bias = Parameter(Tensor(np.zeros((dim))), device=device, dtype=dtype)

        self.running_mean = Tensor(np.zeros((dim)), device=device, dtype=dtype, requires_grad=False)
        self.running_var = Tensor(np.ones((dim)), device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        if self.training:
            mean = ndl.summation(x, axes=(0, )) / n
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            mean = mean.broadcast_to(x.shape)
            var = ndl.summation((x - mean) ** 2, axes=(0, )) / n
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
            var = var.broadcast_to(x.shape)
        else:
            mean = self.running_mean.broadcast_to(x.shape)
            var = self.running_var.broadcast_to(x.shape)

        y = (x - mean) / ((var + self.eps) ** 0.5) * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        return y    

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Parameters
        self.weight = Parameter(Tensor(np.ones((dim))), device=device, dtype=dtype)
        self.bias = Parameter(Tensor(np.zeros((dim))), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        mean = ndl.summation(x, axes=(1, )) / self.dim
        mean = mean.reshape((mean.shape[0], 1)).broadcast_to(x.shape)
        var = ndl.summation((x - mean) ** 2, axes=(1, )) / self.dim
        var = var.reshape((var.shape[0], 1)).broadcast_to(x.shape)
        y = (x - mean) / ((var + self.eps) ** 0.5) * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        return y
    
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p = 1 - self.p)
            x = mask * x / (1 - self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
