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

    printf("%f",C(0, 0));
}

int main(){
    int M = 2048;
    int N = 2048;
    int K = 2048;

    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);

    float *matrix_A_host = (float*)malloc(mem_size_A);
    float *matrix_B_host = (float*)malloc(mem_size_B);

    float *matrix_C_gpu_host = (float*)malloc(mem_size_C);
    float *matrix_C_cpu_host = (float*)malloc(mem_size_C);

    random_matrix(M, N, K, matrix_A_host, matrix_B_host);
    cpu_sgemm(M, N, K, matrix_A_host, matrix_B_host, matrix_C_cpu_host);

    // float *matrix_A_device, matrix_B_device, matrix_C_device;
    // cudaMalloc((void**)&matrix_A_device, mem_size_A);
    // cudaMalloc((void**)&matrix_B_device, mem_size_B);
    // cudaMalloc((void**)&matrix_C_device, mem_size_C);

    // cudaMemcpy(matrix_A_host, matrix_A_device, mem_size_A, cudaMemcpyHostToDevice);
    // cudaMemcpy(matrix_A_host, matrix_A_device, mem_size_A, cudaMemcpyHostToDevice);

    // cudaMemcpy(matrix_A_host, matrix_A_device, mem_size_A, cudaMemcpyHostToDevice);

    return 0;
}