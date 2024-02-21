#include <cuda.h>
#include <iostream>

// block tile
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 8

// element per thread  
#define THREAD_N 8

#define OFFSET(i, j, N) ((i) * (N) + (j)) 
#define FLOAT4(pointer) reinterpret_cast<float4*>(&pointer)[0]
__global__ void matmul(float *a, float *b, float *c, int M, int N, int K){

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


int main(){
    // Problem size
    int M = 128;
    int N = 128;
    int K = 128;
    // Host
    float *a = (float *)malloc(M * K * sizeof(float));
    float *b = (float *)malloc(K * N * sizeof(float));
    float *c = (float *)malloc(M * N * sizeof(float));
    for(int i = 0; i < M * K; i++) 
        a[i] = 1;
    for(int i = 0; i < K * N; i++)
        b[i] = 1;
    
    // Device
    float *da, *db, *dc;
    cudaMalloc(&da, M * K * sizeof(float));
    cudaMalloc(&db, K * N * sizeof(float));
    cudaMalloc(&dc, M * N * sizeof(float));

    // Copy to device
    cudaMemcpy(da, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_N);
    matmul<<<grid, block>>>(da, db, dc, M, N, K);

    // Copy to host
    cudaMemcpy(c, dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++)
            std::cout << c[i * N + j] << " ";
        std::cout << std::endl;
    }

    // Free
    free(a);
    free(b);
    free(c);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}