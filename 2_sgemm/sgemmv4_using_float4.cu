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

#define FETCH_FLOAT4(pointer_val) (reinterpret_cast<float4 *>(&(pointer_val))[0])

template <unsigned int M_PER_BLOCK, 
          unsigned int N_PER_BLOCK,
          unsigned int K_PER_BLOCK,
          unsigned int NUM_PER_THREAD>
__global__ void sgemm(int M, int N, const int K, float* A, float* B, float* C){
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *A_ptr_start = A + M_PER_BLOCK * blockIdx.y * K;
    float *B_ptr_start = B + N_PER_BLOCK * blockIdx.x;
    
    __shared__ float A_shared[M_PER_BLOCK][K_PER_BLOCK];
    __shared__ float B_shared[K_PER_BLOCK][N_PER_BLOCK];
    float temp[NUM_PER_THREAD]={0.f};

    for(int s=0; s<K; s+=K_PER_BLOCK){
        // FETCH_FLOAT4(A_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[s + tx * NUM_PER_THREAD + K * ty]);
        A_shared[ty][tx * NUM_PER_THREAD] = A_ptr_start[s + tx * NUM_PER_THREAD + K * ty];
        A_shared[ty][tx * NUM_PER_THREAD + 1] = A_ptr_start[s + tx * NUM_PER_THREAD + K * ty + 1];
        A_shared[ty][tx * NUM_PER_THREAD + 2] = A_ptr_start[s + tx * NUM_PER_THREAD + K * ty + 2];
        A_shared[ty][tx * NUM_PER_THREAD + 3] = A_ptr_start[s + tx * NUM_PER_THREAD + K * ty + 3];
        // FETCH_FLOAT4(B_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD]);
        B_shared[ty][tx * NUM_PER_THREAD] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD];
        B_shared[ty][tx * NUM_PER_THREAD + 1] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 1];
        B_shared[ty][tx * NUM_PER_THREAD + 2] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 2];
        B_shared[ty][tx * NUM_PER_THREAD + 3] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 3];

        __syncthreads();
        for(int i=0; i<NUM_PER_THREAD; i++){
            for (int k = 0; k < K_PER_BLOCK; k++)
                temp[i] += A_shared[ty][k] * B_shared[k][tx * NUM_PER_THREAD + i];  // 注意 k 是A的列，B的行
        }
        

        __syncthreads();
    }

    float * C_ptr_start = C + (blockIdx.y * M_PER_BLOCK)*N + blockIdx.x * N_PER_BLOCK;
    for (int i=0;i<NUM_PER_THREAD;i++){
        C_ptr_start[N*ty + tx*NUM_PER_THREAD + i] = temp[i];
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

    constexpr int m_blocksize = 32;
    constexpr int n_blocksize = 32;
    constexpr int k_blocksize = 32;
    constexpr int num_thread = 4;

    dim3 Block(8, 32);
    dim3 Grid((M + m_blocksize - 1) / m_blocksize, (N + n_blocksize - 1) / n_blocksize);

    sgemm<m_blocksize, n_blocksize, k_blocksize, num_thread><<<Grid, Block>>>(M, N, K, matrix_A_device, matrix_B_device, matrix_C_device);

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