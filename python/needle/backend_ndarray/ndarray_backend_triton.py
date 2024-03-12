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
def add_kernel(x_ptr, 
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
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, o, num, BLOCK_SIZE=1024)
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
    

@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M, N, K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = N // BLOCK_N
    pid_n = pid % num_pid;
    pid_m = pid // num_pid
    
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    k_offset = tl.arange(0, BLOCK_K)
    
    a_start = a_ptr + (m_offset[:, None] * stride_am + k_offset[None, :] * stride_ak)
    b_start = b_ptr + (k_offset[:, None] * stride_bk + n_offset[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        # load
        a = tl.load(a_start)
        b = tl.load(b_start)
        # dot
        acc += tl.dot(a, b)
        # advanced
        a_start += BLOCK_K * stride_ak
        b_start += BLOCK_K * stride_bk
    
    c = acc.to(tl.float16)
    c_start = out_ptr + (m_offset[:, None] * stride_cm + n_offset[None, :] * stride_cn)
    tl.store(c_start, c)
    

def matmul(a, b, out, m, n, p):
    a = torch.tensor(a.array.reshape(m, n), device='cuda')
    b = torch.tensor(b.array.reshape(n, p), device='cuda')
    # a.array = a.array.reshape(m, n).to('cuda')
    # b.array = b.array.reshape(n, p).to('cuda')  
    # out.array = out.array.reshape(m, p).to('cuda')
    o = torch.tensor(out.array.reshape(m, p), device='cuda')
    
    for BLOCK_M in [2 ** i for i in range(4, 7)]:
        for BLOCK_N in [2 ** i for i in range(4, 7)]:
            for BLOCK_K in [2 ** i for i in range(4, 7)]:
                grid = lambda META: (triton.cdiv(m, META['BLOCK_M']) * triton.cdiv(p, META['BLOCK_N']), )
                matmul_kernel[grid](a, b, o, 
                                    m, p, n, 
                                    a.stride(0), a.stride(1), 
                                    b.stride(0), b.stride(1), 
                                    o.stride(0), o.stride(1),
                                    BLOCK_M, BLOCK_N, BLOCK_K,
                                    )
                out.array = o.cpu()
                print(o)
        


# def matmul(a, b, out, m, n, p):
#     out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)



@triton.jit
def ewise_negative_kernel(
    a_ptr,
    out_ptr,
    num,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    a_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    a = tl.load(a_ptr + a_start)
    
    out = -a
    
    tl.store(out_ptr + out_start, out)
    
    
    
    

def ewise_negative(a, out):
    a = torch.tensor(a.array, device="cuda")
    o = torch.tensor(out.array, device="cuda")
    BLOCK_SIZE = 32
    
    num = len(a)
    
    grid = lambda meta: (triton.cdiv(num, meta['BLOCK_SIZE']), )
    
    ewise_negative_kernel[grid](a, o, num, BLOCK_SIZE)
    out.array = o.cpu()