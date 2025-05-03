#ifndef RADIATOR_GPU_H
#define RADIATOR_GPU_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU内存管理接口
void gpu_alloc_memory(float** d_in, float** d_out, int n, int m);
void gpu_free_memory(float* d_in, float* d_out);

// 数据传输接口
void copy_to_device(float* d_in, const float* h_in, int n, int m);
void copy_from_device(float* h_out, const float* d_out, int n, int m);

// 核心计算接口 (暂为框架)
void gpu_propagate(float* d_in, float* d_out, int n, int m, 
                  int block_x, int block_y, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // RADIATOR_GPU_H
