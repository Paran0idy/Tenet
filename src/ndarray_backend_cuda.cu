#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= size) return ;

  CudaVec idx;
  idx.size = 0;
  size_t index = gid;
  size_t depth = 0;

  while(index){
    idx.data[idx.size++] = index % shape.data[shape.size - 1 - depth];
    index /= shape.data[shape.size - 1 - depth];
    depth++;
  }

  size_t cod = offset;
  for(int i = 0; i < idx.size; i++)
    cod += strides.data[strides.size - 1 - i] * idx.data[i];
  out[gid] = a[cod];
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= size) return ;

  CudaVec idx;
  idx.size = 0;
  size_t index = gid;
  size_t depth = 0;

  while(index){
    idx.data[idx.size++] = index % shape.data[shape.size - 1 - depth];
    index /= shape.data[shape.size - 1 - depth];
    depth++;
  }

  size_t cod = offset;
  for(int i = 0; i < idx.size; i++)
    cod += strides.data[strides.size - 1 - i] * idx.data[i];
  out[cod] = a[gid];
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= size) return ;

  CudaVec idx;
  idx.size = 0;
  size_t index = gid;
  size_t depth = 0;

  while(index){
    idx.data[idx.size++] = index % shape.data[shape.size - 1 - depth];
    index /= shape.data[shape.size - 1 - depth];
    depth++;
  }

  size_t cod = offset;
  for(int i = 0; i < idx.size; i++)
    cod += strides.data[strides.size - 1 - i] * idx.data[i];
  out[cod] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);

}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

// Mul
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// Div
__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {

  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// Power
__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {

  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// Maximum
__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] > val ? a[gid] : val;
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {

  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// Eq
__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == b[gid];
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] == val;
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {

  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// Ge
__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= b[gid];
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] >= val;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {

  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

// log
__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

// Exp
__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

// Tanh
__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t size, uint32_t M, uint32_t N, uint32_t P){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= size) return ;

  size_t row = gid / P, col = gid % P;
  for(int i = 0; i < N; i++)
    out[gid] += a[row * N + i] * b[i * P + col];
}

// block tile
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 8

// element per thread  
#define THREAD_N 8

#define OFFSET(i, j, N) ((i) * (N) + (j)) 
#define FLOAT4(pointer) reinterpret_cast<float4*>(&pointer)[0]
__global__ void matmul_warpTiling(float *a, float *b, float *c, int M, int N, int K){

    __shared__ float shared_a[BLOCK_M][BLOCK_K];
    __shared__ float shared_b[BLOCK_K][BLOCK_N];

    float res[THREAD_N][THREAD_N] = {0};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_m = tid / 2;
    int smem_a_k = (tid % 2) * 4;

    int smem_b_k = tid / 32;
    int smem_b_n = (tid % 32) * 4;

    int gmem_a_m = BLOCK_M * blockIdx.y + smem_a_m;
    int gmem_b_n = BLOCK_N * blockIdx.x + smem_b_n;

    int cta_size[2] = {4, 2};
    int warp_size[2] = {4, 8};
    int WARP_M = BLOCK_M / cta_size[0];
    int WARP_N = BLOCK_N / cta_size[1];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int wy = warp_id / cta_size[1];
    int wx = warp_id % cta_size[1];

    int ty = lane_id / warp_size[1];
    int tx = lane_id % warp_size[1];

    for(int k = 0; k < K / BLOCK_K; k++){
        // GMEM -> SMEM
        int gmem_a_k = k * BLOCK_K + smem_a_k;
        int gmem_b_k = k * BLOCK_K + smem_b_k;

        FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
        __syncthreads();

        // Compute
        for(int kk = 0; kk < BLOCK_K; kk++)
            for(int i = 0; i < 2; i++){
                int thread_y = wy * WARP_M + THREAD_N / 2 * ty + i * WARP_M / 2;
                for(int j = 0; j < 2; j++){
                    int thread_x = wx * WARP_N + THREAD_N / 2 * tx + j * WARP_N / 2;
                    for(int m = 0; m < THREAD_N / 2; m++)
                        for(int n = 0; n < THREAD_N / 2; n++)
                            res[i * THREAD_N / 2 + m][j * THREAD_N / 2 + n] += shared_a[thread_y + m][kk] * shared_b[kk][thread_x + n];
                }
            }
        __syncthreads();
    }

    // Write back
    int block_y = BLOCK_M * blockIdx.y;
    int block_x = BLOCK_N * blockIdx.x;  
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++){
            int thread_y = wy * WARP_M + THREAD_N / 2 * ty + i * WARP_M / 2;
            int thread_x = wx * WARP_N + THREAD_N / 2 * tx + j * WARP_N / 2;
            for(int m = 0; m < THREAD_N / 2; m++)
                for(int n = 0; n < THREAD_N / 2; n++)
                    c[OFFSET(block_y + thread_y + m, block_x + thread_x + n, N)] = res[i * THREAD_N / 2 + m][j * THREAD_N / 2 + n];
        }
}


// /*
// Thread Tiling
// */

// // block tile
// #define BLOCK_M 128
// #define BLOCK_N 128
// #define BLOCK_K 8

// // element per thread  
// #define THREAD_N 8

// #define OFFSET(i, j, N) ((i) * (N) + (j)) 
// #define FLOAT4(pointer) reinterpret_cast<float4*>(&pointer)[0]
// __global__ void matmul_threadTiling(float *a, float *b, float *c, int M, int N, int K){

//     __shared__ float shared_a[BLOCK_M][BLOCK_K];
//     __shared__ float shared_b[BLOCK_K][BLOCK_N];

//     float res[THREAD_N][THREAD_N] = {0};

//     int tid = threadIdx.y * blockDim.x + threadIdx.x;

//     int smem_a_m = tid / 2;
//     int smem_a_k = (tid % 2) * 4;

//     int smem_b_k = tid / 32;
//     int smem_b_n = (tid % 32) * 4;

//     int gmem_a_m = BLOCK_M * blockIdx.y + smem_a_m;
//     int gmem_b_n = BLOCK_N * blockIdx.x + smem_b_n;

//     for(int k = 0; k < K / BLOCK_K; k++){
//         // GMEM -> SMEM
//         int gmem_a_k = k * BLOCK_K + smem_a_k;
//         int gmem_b_k = k * BLOCK_K + smem_b_k;

//         FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
//         FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
//         __syncthreads();

//         // Compute
//         int ty = threadIdx.y * THREAD_N;
//         int tx = threadIdx.x * THREAD_N;
//         for(int kk = 0; kk < BLOCK_K; kk++)
//             for(int ii = 0; ii < THREAD_N; ii++)
//                 for(int jj = 0; jj < THREAD_N; jj++)
//                     res[ii][jj] += shared_a[ty + ii][kk] * shared_b[kk][tx + jj];
//         __syncthreads();
//     }

//     // Write back
//     int ty = BLOCK_M * blockIdx.y + THREAD_N * threadIdx.y;
//     int tx = BLOCK_N * blockIdx.x + THREAD_N * threadIdx.x;

//     for(int i = 0; i < THREAD_N; i++)
//         for(int j = 0; j < THREAD_N; j++)
//             c[OFFSET(ty + i, tx + j, N)] = res[i][j];
// }

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  // Thread tiling
  if(M % BLOCK_M == 0 && N % BLOCK_K == 0 && P % BLOCK_N == 0){
    dim3 grid(P / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_N);
    matmul_warpTiling<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, P, N);
    return ;
  }

  CudaDims dim = CudaOneDim(M * N * P);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M * P, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, uint32_t reduce_size, uint32_t out_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(gid >= out_size) return ;
  float local_max = a[gid * reduce_size];
  for(int i = gid * reduce_size; i < (gid + 1) * reduce_size; i++)
    local_max = max(a[i], local_max);
  out[gid] = local_max;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, uint32_t reduce_size, uint32_t out_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(gid >= out_size) return ;
  float local_sum = 0;
  for(int i = gid * reduce_size; i < (gid + 1) * reduce_size; i++)
    local_sum += a[i];
  out[gid] = local_sum;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}


// EwiseNegative

__global__ void EwiseNegativeKernel(const scalar_t *a, scalar_t *out, uint32_t out_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= out_size) return ;

  out[gid] = -out[gid];
}

void EwiseNegative(const CudaArray &a, CudaArray *out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseNegativeKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("ewise_negative", EwiseNegative);
}
