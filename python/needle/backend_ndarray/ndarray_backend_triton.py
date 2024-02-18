import numpy as np
import triton
import triton.language as tl
import torch


__device_name__ = "triton"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size):
        self.array = np.empty(size, dtype=np.float32)

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
    )


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


@triton.jit
def triton_add(x_ptr, 
               y_ptr , 
               out_ptr, 
               num,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offset = start + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(x_ptr + offset)
    y = tl.load(y_ptr + offset)
    
    out = x + y
        
    tl.store(out_ptr + offset, out)
    

def ewise_add(a, b, out):
    x = torch.tensor(a.array, device='cuda')
    y = torch.tensor(b.array, device='cuda')
    o = torch.empty_like(x)
    num = o.size()[0]
    print(num)
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK_SIZE']), )
    triton_add[grid](x, y, o, num, BLOCK_SIZE=2)
    out.array[:] = o.cpu().numpy()
    

def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)
