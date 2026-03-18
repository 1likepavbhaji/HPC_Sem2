
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <iostream>

#define N 2048

__global__
void matMul(const int* a, const int* b, int* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;

        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }

        c[row * N + col] = sum;
    }
}

int main(void) {

    int* arr1 = (int*) malloc(sizeof(int) * N * N);
    int* arr2 = (int*) malloc(sizeof(int) * N * N);
    int* arr3 = (int*) malloc(sizeof(int) * N * N);
    int* arrt = (int*) malloc(sizeof(int) * N * N);

    int* cuda_p1;
    int* cuda_p2;
    int* cuda_p3;

    // initialize
    for (int i = 0; i < N * N; i++) {
        arr1[i] = 1;
        arr2[i] = 2;
    }

    cudaMalloc((void**)&cuda_p1, sizeof(int) * N * N);
    cudaMalloc((void**)&cuda_p2, sizeof(int) * N * N);
    cudaMalloc((void**)&cuda_p3, sizeof(int) * N * N);

    cudaMemcpy(cuda_p1, arr1, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_p2, arr2, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);

    auto startcuda = std::chrono::high_resolution_clock::now();

    matMul<<<blocksPerGrid, threadsPerBlock>>>(cuda_p1, cuda_p2, cuda_p3);

    cudaDeviceSynchronize();

    cudaMemcpy(arr3, cuda_p3, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    auto endcuda = std::chrono::high_resolution_clock::now();


    // CPU multiply

    auto startcpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            int sum = 0;

            for (int k = 0; k < N; k++) {
                sum += arr1[i*N + k] * arr2[k*N + j];
            }

            arrt[i*N + j] = sum;
        }
    }

    auto endcpu = std::chrono::high_resolution_clock::now();


    double cuda_time =
    std::chrono::duration<double>(endcuda - startcuda).count();

    double cpu_time =
    std::chrono::duration<double>(endcpu - startcpu).count();


    std::cout << cuda_time << " sec on GPU vs "
              << cpu_time << " sec on CPU " << std::endl;

    std::cout << cpu_time / cuda_time << std::endl;

    return 0;
}
