"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray, array_api
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks



class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        input = node.inputs
        return out_grad * self.scalar * power_scalar(input[0], self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        grad_a = out_grad / rhs
        grad_b = - out_grad * lhs / (rhs ** 2)
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axis = list(range(len(a.shape)))
        if self.axes is not None:
            axis[self.axes[0]], axis[self.axes[1]] = axis[self.axes[1]], axis[self.axes[0]]
            return array_api.transpose(a, axis)
        
        axis[-1], axis[-2] = axis[-2], axis[-1]
        return array_api.transpose(a, axis)

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        input = node.inputs
        return reshape(out_grad, shape=input[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, shape=self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        
        reduce_dim = list()
        # [i : broadcast idx, j : input idx] from right to left
        j = len(input_shape) - 1
        for i in range(len(self.shape) - 1, -1, -1):
            
            if j < 0:
                reduce_dim.append(i)
                continue
            
            origin = input_shape[j]
            broadcast = self.shape[i]
            
            # must a broadcast dim
            if origin != broadcast:
                reduce_dim.append(i)
            j -= 1
        
        return reshape(summation(out_grad, tuple(reduce_dim)), input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        shape = list(input_shape)
        
        if self.axes:
            # if self.axes is isinstance(self.axes, int):
            #     self.axes = (self.axes, )
            for i in list(self.axes):
                shape[i] = 1
        else:
            shape = [1 for _ in range(len(shape))]
            out_grad = Tensor(out_grad.cached_data * array_api.ones(input_shape))
            return out_grad

        return reshape(out_grad, shape=shape) * array_api.ones(input_shape)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        
        grad_a = matmul(out_grad, transpose(rhs))
        grad_b = matmul(transpose(lhs), out_grad)
        
        if grad_a.shape != lhs.shape:
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape) - len(lhs.shape))))
        if grad_b.shape != rhs.shape:
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape) - len(rhs.shape))))
        
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        input = node.inputs
        return out_grad / input[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        input = node.inputs
        return out_grad * exp(input[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return a * (a > 0)

    def gradient(self, out_grad, node):
        input = node.inputs
        return out_grad * (input[0].realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)
