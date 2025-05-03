#include "radiator_gpu.h"
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error (" << __FILE__ << ":" << __LINE__ << "): " \
                  << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA Error"); \
    } \
}

void gpu_alloc_memory(float** d_in, float** d_out, int n, int m) {
    const size_t bytes = n * m * sizeof(float);
    CHECK_CUDA(cudaMalloc(d_in, bytes));
    CHECK_CUDA(cudaMalloc(d_out, bytes));
    CHECK_CUDA(cudaMemset(*d_in, 0, bytes));
    CHECK_CUDA(cudaMemset(*d_out, 0, bytes));
}

void gpu_free_memory(float* d_in, float* d_out) {
    if(d_in) CHECK_CUDA(cudaFree(d_in));
    if(d_out) CHECK_CUDA(cudaFree(d_out));
}

void copy_to_device(float* d_in, const float* h_in, int n, int m) {
    CHECK_CUDA(cudaMemcpy(d_in, h_in, n*m*sizeof(float), 
                        cudaMemcpyHostToDevice));
}

void copy_from_device(float* h_out, const float* d_out, int n, int m) {
    CHECK_CUDA(cudaMemcpy(h_out, d_out, n*m*sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void gpu_propagate(float* d_in, float* d_out, int n, int m,
                  int block_x, int block_y, cudaStream_t stream) {
    // 参数验证
    if (m % block_x != 0 || n % block_y != 0) {
        throw std::invalid_argument("Block size must divide matrix dimensions");
    }

    // 空内核框架（后续实现）
    dim3 blocks((m + block_x - 1)/block_x, (n + block_y - 1)/block_y);
    dim3 threads(block_x, block_y);
    
    // 实际内核调用将在此处添加
    // propagate_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n, m);
    
// CHECK_CUDA(cudaGetLastError());
}
