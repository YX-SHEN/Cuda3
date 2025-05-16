#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// GPU device 端算法完全等价CPU
__device__ double expint_double_dev(int n, double x, int maxIterations) {
    const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30, bigDouble = 1.7976931348623157e+308;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;
    if (n <= 0 || x < 0 || (x == 0.0 && (n == 0 || n == 1))) return -1.0;
    if (n == 0) return exp(-x) / x;
    if (x > 1.0) {
        b = x + n; c = bigDouble; d = 1.0 / b; h = d;
        for (i = 1; i <= maxIterations; i++) {
            a = -i * (nm1 + i); b += 2.0; d = 1.0 / (a * d + b);
            c = b + a / c; del = c * d; h *= del;
            if (fabs(del - 1.0) <= epsilon) return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant); fact = 1.0;
        for (i = 1; i <= maxIterations; i++) {
            fact *= -x / i;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon) return ans;
        }
        return ans;
    }
}
__device__ float expint_float_dev(int n, float x, int maxIterations) {
    const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f, bigfloat = 3.402823466e+38f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;
    if (n <= 0 || x < 0 || (x == 0.0f && (n == 0 || n == 1))) return -1.0f;
    if (n == 0) return expf(-x) / x;
    if (x > 1.0f) {
        b = x + n; c = bigfloat; d = 1.0f / b; h = d;
        for (i = 1; i <= maxIterations; i++) {
            a = -i * (nm1 + i); b += 2.0f; d = 1.0f / (a * d + b);
            c = b + a / c; del = c * d; h *= del;
            if (fabsf(del - 1.0f) <= epsilon) return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant); fact = 1.0f;
        for (i = 1; i <= maxIterations; i++) {
            fact *= -x / i;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
    }
}

// GPU kernel，每个线程计算 (i, j) 一个结果
__global__ void expint_kernel(int n, int m, double a, double b, int maxIterations,
                              float *resultsFloat, double *resultsDouble) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n * m;
    if (tid < total) {
        int i = tid / m;      // order index
        int j = tid % m;      // sample index
        double x = a + (j + 1) * (b - a) / ((double)m);
        int order = i + 1;
        resultsDouble[tid] = expint_double_dev(order, x, maxIterations);
        resultsFloat[tid]  = expint_float_dev(order, (float)x, maxIterations);
    }
}

// 主机接口
void expint_cuda(int n, int m, double a, double b, int maxIterations,
                 std::vector<std::vector<float>> &gpuFloat, std::vector<std::vector<double>> &gpuDouble,
                 float &gpuTimeMs)
{
    int total = n * m;
    float *d_resultsFloat = nullptr;
    double *d_resultsDouble = nullptr;
    cudaMalloc(&d_resultsFloat, total * sizeof(float));
    cudaMalloc(&d_resultsDouble, total * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    expint_kernel<<<numBlocks, blockSize>>>(n, m, a, b, maxIterations, d_resultsFloat, d_resultsDouble);
    cudaDeviceSynchronize();

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);

    // host 结果收集
    std::vector<float> h_resultsFloat(total);
    std::vector<double> h_resultsDouble(total);
    cudaMemcpy(h_resultsFloat.data(), d_resultsFloat, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resultsDouble.data(), d_resultsDouble, total * sizeof(double), cudaMemcpyDeviceToHost);

    // reshape 到二维
    gpuFloat.resize(n, std::vector<float>(m));
    gpuDouble.resize(n, std::vector<double>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            gpuFloat[i][j] = h_resultsFloat[i * m + j];
            gpuDouble[i][j] = h_resultsDouble[i * m + j];
        }

    cudaFree(d_resultsFloat);
    cudaFree(d_resultsDouble);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
