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

template <unsigned int BLOCK_SIZE, unsigned int K_>
__global__ void sgemm(int M, int N, const int K, float* A, float* B, float* C){

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    float *A_ptr_start = A + blockDim.y * blockIdx.y * K;
    float *B_ptr_start = B + blockDim.x * blockIdx.x;

    __shared__ float A_shared[BLOCK_SIZE][K_];
    __shared__ float B_shared[K_][BLOCK_SIZE];

    for(int s=0; s<K; s+=blockDim.x){
        A_shared[threadIdx.y][threadIdx.x + s] = A_ptr_start[threadIdx.x + s + threadIdx.y * K];
        B_shared[threadIdx.y + s][threadIdx.x] = B_ptr_start[(threadIdx.y + s)*N + threadIdx.x];
    }
    __syncthreads();

    float temp=0.f;
    for (int k = 0; k < K; k++)
    {
        temp += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];  // 注意 k 是A的列，B的行
    }
    C[x + y * N] = temp;
}

int main(){
    int M = 128;
    int N = 128;
    constexpr int K = 128;

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
    dim3 Block(blocksize, blocksize);
    dim3 Grid((M + blocksize - 1) / blocksize, (N + blocksize - 1) / blocksize);

    sgemm<blocksize, K><<<Grid, Block>>>(M, N, K, matrix_A_device, matrix_B_device, matrix_C_device);

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