#include "expint_gpu.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace gpu {

// 内存分配/释放
void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples) {
    size_t bytes = samples * sizeof(float);
    cudaMalloc((void**)&d_x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
}

void alloc_and_copy_to_device(const double* h_x, double*& d_x, int samples) {
    size_t bytes = samples * sizeof(double);
    cudaMalloc((void**)&d_x, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
}

void free_device(float* d_x) { if (d_x) cudaFree(d_x); }
void free_device(double* d_x) { if (d_x) cudaFree(d_x); }

// ==================== device-side precise En(x) ====================
template <typename T>
__device__ T d_expint_impl(int n, T x) {
    const T euler = (sizeof(T) == sizeof(double)) ? T(0.5772156649015328606) : T(0.5772156649f);
    const T eps = 1e-30;
    const T big = (sizeof(T) == sizeof(double)) ? T(1.0e308) : T(3.4e38f);
    const int maxIter = 10000;

    if (n == 0) return exp(-x) / x;

    int nm1 = n - 1;
    if (x > 1.0) {
        T b = x + n, c = big, d = 1.0 / b, h = d;
        for (int i = 1; i <= maxIter; ++i) {
            T a = -i * (nm1 + i);
            b += 2;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            T del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= eps) return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        T ans = (nm1 ? T(1.0) / nm1 : -log(x) - euler);
        T fact = 1.0;
        for (int i = 1; i <= maxIter; ++i) {
            fact *= -x / i;
            T del;
            if (i != nm1)
                del = -fact / (i - nm1);
            else {
                T psi = -euler;
                for (int k = 1; k <= nm1; ++k) psi += T(1.0) / k;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * eps) return ans;
        }
        return ans;
    }
}

// ==================== multi-n kernel ====================
template <typename T>
__global__ void expint_kernel_multi(int nMax, const T* x, T* out, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= samples) return;
    T xi = x[idx];
    for (int n = 0; n <= nMax; ++n) {
        out[idx * (nMax + 1) + n] = d_expint_impl<T>(n, xi);
    }
}

// ==================== dual-stream overlap launcher ====================
template <typename T>
void expint_gpu_multi_stream(int nMax, const T* d_x, T* d_out, int samples, int blockSize) {
    const int numStreams = 2;
    int chunk = (samples + numStreams - 1) / numStreams;

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i)
        cudaStreamCreate(&streams[i]);

    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunk;
        if (offset >= samples) break;
        int len = std::min(chunk, samples - offset);

        const T* x_ptr = d_x + offset;
        T* out_ptr = d_out + offset * (nMax + 1);

        int grid = (len + blockSize - 1) / blockSize;
        expint_kernel_multi<T><<<grid, blockSize, 0, streams[i]>>>(nMax, x_ptr, out_ptr, len);
    }

    for (int i = 0; i < numStreams; ++i)
        cudaStreamSynchronize(streams[i]);

    for (int i = 0; i < numStreams; ++i)
        cudaStreamDestroy(streams[i]);
}

// ==================== public API ====================
void expint_gpu_multi_float(int nMax, const float* d_x, float* d_out, int samples, int blockSize) {
    expint_gpu_multi_stream<float>(nMax, d_x, d_out, samples, blockSize);
}

void expint_gpu_multi_double(int nMax, const double* d_x, double* d_out, int samples, int blockSize) {
    expint_gpu_multi_stream<double>(nMax, d_x, d_out, samples, blockSize);
}

} // namespace gpu
