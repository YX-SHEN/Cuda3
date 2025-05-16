#pragma once

#ifndef COMPILE_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace gpu {

#ifndef COMPILE_CPU_ONLY

// ----------- 内存管理 ----------
void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples);
void alloc_and_copy_to_device(const double* h_x, double*& d_x, int samples);
void free_device(float* d_x);
void free_device(double* d_x);

// ----------- kernel 启动器（旧接口） ----------
void expint_gpu_float (const int n, const float*  d_x, float*  d_out, int samples, int blockSize);
void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples, int blockSize);

inline void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples) {
    expint_gpu_float(n, d_x, d_out, samples, 128);
}
inline void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples) {
    expint_gpu_double(n, d_x, d_out, samples, 128);
}

// ----------- 一次计算 (n+1) 个 Eₙ ----------
void expint_gpu_multi_float (int nMax, const float*  d_x, float*  d_out, int samples, int blockSize);
void expint_gpu_multi_double(int nMax, const double* d_x, double* d_out, int samples, int blockSize);

// ----------- 支持 Dual-Stream 重叠 ----------
void expint_float_array (int nMax, const float*  xIn, float*  out, int samples);
void expint_double_array(int nMax, const double* xIn, double* out, int samples);

// ----------- 新增：Dynamic Parallelism 支持 ----------
void expint_gpu_dp_float(int nMax, const float* h_x, float* h_out, int samples);

#endif

} // namespace gpu
