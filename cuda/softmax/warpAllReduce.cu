#include <cuda.h>
#include <iostream>

__device__ float warpAllRduce_max(float val){
    for(int i = 16; i > 0; i /= 2)
        val = max(__shfl_xor_sync(0xffffffff, val, i, 32), val);
    return val;
}

__device__ float warpAllRduce_sum(float val){
    for(int i = 16; i > 0; i /= 2)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    return val;
}

#define OFFSET(i, j, N) (i) * (N) + (j)
__global__ void softmax(float *a, float *b, int M, int N){
    for(int r = blockIdx.x * blockDim.y; r < M; r += gridDim.x * blockDim.y){
        int i = threadIdx.y;
        float local_max = 0;
        float local_sum = 0;
        // max
        for(int c = threadIdx.x; c < N / 4; c += blockDim.x)
            for(int j = 0; j < 4; j++)
                local_max = max(a[OFFSET(r + i, c * 4 + j, N)], local_max);
        __syncthreads();
        local_max = warpAllRduce_max(local_max);

        // exp(val - max)
        for(int c = threadIdx.x; c < N / 4; c += blockDim.x)
            for(int j = 0; j < 4; j++)
                a[OFFSET(r + i, c * 4 + j, N)] = exp(a[OFFSET(r + i, c * 4 + j, N)] - local_max);
        __syncthreads();

        // sum
        for(int c = threadIdx.x; c < N / 4; c += blockDim.x)
            for(int j = 0; j < 4; j++)
                local_sum += a[OFFSET(r + i, c * 4 + j, N)];
        __syncthreads();
        local_sum = warpAllRduce_sum(local_sum);

        // val / sum
        for(int c = threadIdx.x; c < N / 4; c += blockDim.x)
            for(int j = 0; j < 4; j++)
                a[OFFSET(r + i, c * 4 + j, N)] /= local_sum;
        __syncthreads();
    }
}


int main(){
    int M = 4, N = 128;
    float *a = new float[M * N];
    float *b = new float[M * N];
    for(int i = 0; i < M * N; i++) 
        a[i] = 1;

    float *da, *db;
    cudaMalloc(&da, M * N * sizeof(float));
    cudaMalloc(&db, M * N * sizeof(float));
    cudaMemcpy(da, a, M * N * sizeof(float), cudaMemcpyHostToDevice);

    softmax<<<2, dim3(32, 2)>>>(da, db, M, N);

    cudaMemcpy(a, da, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++)
            std::cout << a[i * N + j] << " ";
        std::cout << std::endl;
        std::cout << std::endl;
    }

    delete []a;
    delete []b;

    cudaFree(da);
    cudaFree(db);

    return 0;
}