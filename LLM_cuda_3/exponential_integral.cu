#include <cuda_runtime.h>
#include <cmath>

// Euler's constant stored in constant memory
__constant__ double d_eulerConstant = 0.5772156649015329;
__constant__ float  d_eulerConstantF = 0.5772156649015329f;

__device__ double exponentialIntegralDouble(const int n, const double x, const int maxIterations) {
    const double epsilon = 1.E-30;
    const double bigDouble = 1.7976931348623157e+308;
    int nm1 = n - 1;
    double ans = 0.0;

    if (n <= 0 || x < 0 || (x == 0.0 && (n == 0 || n == 1))) return -1.0;

    if (n == 0) return exp(-x) / x;
    
    if (x > 1.0) {
        double b = x + n;
        double c = bigDouble;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= maxIterations; i++) {
            double a = -double(i) * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            double del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon) return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0) ? 1.0 / nm1 : (-log(x) - d_eulerConstant);
        double fact = 1.0;
        for (int i = 1; i <= maxIterations; i++) {
            fact *= -x / i;
            double del = (i != nm1) ? -fact / (i - nm1) : fact * (-log(x) + d_eulerConstant);
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__device__ float exponentialIntegralFloat(const int n, const float x, const int maxIterations) {
    const float epsilon = 1.E-30f;
    const float bigFloat = 3.402823466e+38F;
    int nm1 = n - 1;
    float ans = 0.0f;

    if (n <= 0 || x < 0 || (x == 0.0f && (n == 0 || n == 1))) return -1.0f;

    if (n == 0) return expf(-x) / x;
    
    if (x > 1.0f) {
        float b = x + n;
        float c = bigFloat;
        float d = 1.0f / b;
        float h = d;
        for (int i = 1; i <= maxIterations; i++) {
            float a = -float(i) * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            float del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= epsilon) return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0) ? 1.0f / nm1 : (-logf(x) - d_eulerConstantF);
        float fact = 1.0f;
        for (int i = 1; i <= maxIterations; i++) {
            fact *= -x / i;
            float del = (i != nm1) ? -fact / (i - nm1) : fact * (-logf(x) + d_eulerConstantF);
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__global__ void computeExponentialIntegral(const int n, const int maxIterations,
                                           const double* x_vals, double* resultsDouble, float* resultsFloat,
                                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        resultsDouble[idx] = exponentialIntegralDouble(n, x_vals[idx], maxIterations);
        resultsFloat[idx]  = exponentialIntegralFloat(n, (float)x_vals[idx], maxIterations);
    }
}
