#include <cstdio>
#include "sgemm_utils.h"
#include <chrono>

void random_matrix(int m, int n, int k, float* A, float* B){
    int lda = k;
    int ldb = n;
    // A matrix init
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++){
            A(i, j) = 2.0 * drand48() - 1.0;
            // A(i, j) = 1.0;
        }
    }
    // B matrix init
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++){
            B(i, j) = 2.0 * drand48() - 1.0;
            // B(i, j) = 1.0;
        }
    }
}

float compare_matrices(int M, int N, float * A, float * B){
    float max_diff = 0.f , diff;
    for (int i = 0; i < M;i++){
        for (int j = 0; j < N;j++){
            diff = abs(A[i*N+j] - B[i*N+j]);
            max_diff = (diff > max_diff ? diff : max_diff);
        }
    }
    printf("max diff = %f \n", max_diff);
}

void cpu_sgemm(int M, int N, int K, float* A, float* B, float* C){
    int lda = K;
    int ldb = N;
    int ldc = N;

    for (int m = 0; m < M; m++){
        for (int n = 0; n < N;n++){
            float temp = 0.f;
            for (int k = 0; k < K; k++)
            {
                temp += A(m,k) * B(k,n);
            }
            C(m, n) = temp;
        }
    }

    printf("%f\n",C(0, 0));
}

template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void sgemm(int M, int N, const int K, float* A, float* B, float* C){
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *A_ptr_start = A + STEP * blockIdx.y * K;
    float *B_ptr_start = B + STEP * blockIdx.x;
    
    __shared__ float A_shared[STEP][STEP];
    __shared__ float B_shared[STEP][STEP];
    float temp[STRIDE][STRIDE]={0.f};

    for(int s=0; s<K; s+=STEP){
        for (int i=0;i<STRIDE;i++){
            for (int j=0;j<STRIDE;j++){
                A_shared[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] = A_ptr_start[s + tx + BLOCK_SIZE * j + K * (ty + BLOCK_SIZE * i)];
                B_shared[ty + BLOCK_SIZE * i][tx + BLOCK_SIZE * j] = B_ptr_start[(ty + BLOCK_SIZE * i + s) * N + tx + BLOCK_SIZE * j];
            }
        }
        __syncthreads();
        for (int i=0;i<STRIDE;i++){
            for (int j=0;j<STRIDE;j++){
                for (int k = 0; k < STEP; k++)
                    temp[i][j] += A_shared[ty + BLOCK_SIZE * i][k] * B_shared[k][tx + BLOCK_SIZE * j];  // 注意 k 是A的列，B的行
            }
        }
        __syncthreads();
    }

    float * C_ptr_start = C + (blockIdx.y * STEP)*N + blockIdx.x * STEP;
    for (int i=0;i<STRIDE;i++){
        for (int j=0;j<STRIDE;j++){
            C_ptr_start[N*(ty + i*BLOCK_SIZE) + tx + j*BLOCK_SIZE] = temp[i][j];
        }
    }

}

int main(){
    int M = 1024;
    int N = 1024;
    constexpr int K = 1024;

    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);

    float *matrix_A_host = (float*)malloc(mem_size_A);
    float *matrix_B_host = (float*)malloc(mem_size_B);

    float *matrix_C_gpu_host = (float*)malloc(mem_size_C);
    float *matrix_C_cpu_host = (float*)malloc(mem_size_C);

    random_matrix(M, N, K, matrix_A_host, matrix_B_host);
    cpu_sgemm(M, N, K, matrix_A_host, matrix_B_host, matrix_C_cpu_host);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void**)&matrix_A_device, mem_size_A);
    cudaMalloc((void**)&matrix_B_device, mem_size_B);
    cudaMalloc((void**)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    constexpr int blocksize = 16;
    constexpr int stride = 2;
    dim3 Block(blocksize, blocksize);
    dim3 Grid((M + blocksize - 1) / blocksize / stride, (N + blocksize - 1) / blocksize / stride);

    sgemm<blocksize, stride><<<Grid, Block>>>(M, N, K, matrix_A_device, matrix_B_device, matrix_C_device);

    cudaMemcpy(matrix_C_gpu_host, matrix_C_device,  mem_size_C, cudaMemcpyDeviceToHost);

    printf("%f\n",matrix_C_gpu_host[0]);
    compare_matrices(M, N, matrix_C_cpu_host, matrix_C_gpu_host);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_cpu_host);
    free(matrix_C_gpu_host);
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);

    return 0;
}