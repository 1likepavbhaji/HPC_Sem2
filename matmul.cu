#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 4

// GPU kernel
__global__ void matmul_gpu(int *A, int *B, int *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU version
void matmul_cpu(int *A, int *B, int *C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    int A[N][N], B[N][N], C[N][N];

    int *d_A, *d_B, *d_C;

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    // Allocate GPU memory
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    // Copy to GPU
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(1, 1);

    matmul_gpu<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result Matrix:\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}