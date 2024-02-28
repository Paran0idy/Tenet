from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_z = array_api.exp(Z - array_api.broadcast_to(z_max, Z.shape))
        return array_api.log(array_api.sum(exp_z, axis=self.axes)) + array_api.sum(z_max, axis=self.axes)
    
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        max_x = array_api.max(x.cached_data, axis=self.axes, keepdims=True)
        exp_x = array_api.exp(x.cached_data - max_x)
        div_x = array_api.sum(exp_x, axis=self.axes, keepdims=True)
        
        input_shape = node.inputs[0].shape
        shape = list(input_shape)
        
        if self.axes:
            if self.axes is isinstance(self.axes, int):
                self.axes = (self.axes, )
            for i in list(self.axes):
                shape[i] = 1
        else:
            shape = [1 for _ in range(len(shape))]
            
        sum_x = reshape(out_grad, shape=shape) * array_api.ones(input_shape)

        return sum_x * exp_x / div_x


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

