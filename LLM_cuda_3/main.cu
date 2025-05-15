// main.cpp
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

__global__ void computeExponentialIntegral(int n, int maxIterations,
                                           const double* x_vals, double* resultsDouble, float* resultsFloat,
                                           int size);

int main() {
    int n = 10;
    int samples = 10;
    int maxIterations = 1000;
    double a = 0.1, b = 10.0;

    // 分配 pinned host 内存（提高 H2D/D2H 速度）
    double *h_x_vals, *h_resultsDouble;
    float  *h_resultsFloat;
    cudaMallocHost((void**)&h_x_vals, samples * sizeof(double));
    cudaMallocHost((void**)&h_resultsDouble, samples * sizeof(double));
    cudaMallocHost((void**)&h_resultsFloat, samples * sizeof(float));

    // 初始化输入
    for (int i = 0; i < samples; i++)
        h_x_vals[i] = a + (b - a) * (double(i) / samples);

    // 分配 device 内存
    double *d_x_vals, *d_resultsDouble;
    float  *d_resultsFloat;
    cudaMalloc((void**)&d_x_vals, samples * sizeof(double));
    cudaMalloc((void**)&d_resultsDouble, samples * sizeof(double));
    cudaMalloc((void**)&d_resultsFloat, samples * sizeof(float));

    cudaMemcpy(d_x_vals, h_x_vals, samples * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (samples + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 kernel
    computeExponentialIntegral<<<blocksPerGrid, threadsPerBlock>>>(
        n, maxIterations, d_x_vals, d_resultsDouble, d_resultsFloat, samples
    );
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_resultsDouble, d_resultsDouble, samples * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resultsFloat, d_resultsFloat, samples * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出前10个结果
    std::cout << std::fixed << std::setprecision(7);
    for (int i = 0; i < samples; i++) {
        std::cout << "x = " << h_x_vals[i]
                  << " | E_" << n << "(x) double = " << h_resultsDouble[i]
                  << " | float = " << h_resultsFloat[i] << std::endl;
    }

    cudaFree(d_x_vals); cudaFree(d_resultsDouble); cudaFree(d_resultsFloat);
    cudaFreeHost(h_x_vals); cudaFreeHost(h_resultsDouble); cudaFreeHost(h_resultsFloat);

    return 0;
}
