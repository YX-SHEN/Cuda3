// File: src/expint_gpu_dp.cu

#include "expint_gpu.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace gpu {

/* =======================================================
   Device-side precise En(x) function (float version)
   ======================================================= */
__device__ float d_expint_impl_float(int n, float x) {
    const float euler = 0.5772156649f;
    const float eps = 1e-30f;
    const int maxIter = 10000;
    if (n == 0) return expf(-x) / x;
    int nm1 = n - 1;
    if (x > 1.0f) {
        float b = x + n, c = 3.4e38f, d = 1.0f / b, h = d;
        for (int i = 1; i <= maxIter; ++i) {
            float a = -i * (nm1 + i);
            b += 2;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            float del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= eps) return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        float ans = (nm1 ? 1.0f / nm1 : -logf(x) - euler);
        float fact = 1.0f;
        for (int i = 1; i <= maxIter; ++i) {
            fact *= -x / i;
            float del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                float psi = -euler; for (int k = 1; k <= nm1; ++k) psi += 1.0f / k;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * eps) return ans;
        }
        return ans;
    }
}

/* =======================================================
   Sub-kernel: Each thread computes E_n for given x
   ======================================================= */
__global__ void expint_subkernel(int nMax, float x, float* out) {
    int n = threadIdx.x;
    if (n > nMax) return;
    out[n] = d_expint_impl_float(n, x);
}

/* =======================================================
   Launcher kernel (Dynamic Parallelism): One per x[i]
   ======================================================= */
__global__ void expint_launcher_dp(int nMax, const float* x, float* out, int samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= samples) return;
    float xi = x[i];
    float* out_i = out + i * (nMax + 1);
    expint_subkernel<<<1, nMax + 1>>>(nMax, xi, out_i);
}

/* =======================================================
   Internal device-level kernel launch
   ======================================================= */
void expint_gpu_dp_float_device(int nMax, const float* d_x, float* d_out, int samples, int blockSize) {
    int block = (blockSize > 0) ? blockSize : 128;
    int grid = (samples + block - 1) / block;
    expint_launcher_dp<<<grid, block>>>(nMax, d_x, d_out, samples);
    cudaDeviceSynchronize();
}

/* =======================================================
   Host wrapper for float (entry from main.cpp)
   ======================================================= */
void expint_gpu_dp_float(int nMax, const float* h_x, float* h_out, int samples) {
    float* d_x = nullptr;
    float* d_out = nullptr;
    size_t size_in  = samples * sizeof(float);
    size_t size_out = samples * (nMax + 1) * sizeof(float);

    cudaMalloc((void**)&d_x,    size_in);
    cudaMalloc((void**)&d_out,  size_out);
    cudaMemcpy(d_x, h_x, size_in, cudaMemcpyHostToDevice);

    expint_gpu_dp_float_device(nMax, d_x, d_out, samples, 128);

    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_out);
}

} // namespace gpu
