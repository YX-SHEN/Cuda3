//------------------------------------------------------------------------------
// File : main.cpp   (with COMPILE_CPU_ONLY + dynamic parallelism support)
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>

#include "include/expint_cpu.hpp"
#ifndef COMPILE_CPU_ONLY
#include "include/expint_gpu.hpp"
#endif

using namespace std;

/* ---------- global flags / params ---------- */
bool verbose = false;
bool timing  = false;
bool cpu_on  = true;
bool gpu_on  = true;
bool use_dp  = false;

unsigned int n             = 10;
unsigned int samples       = 10;
double       a = 0.0, b = 10.0;
int          blockSize     = 128;

inline double nowSeconds() {
    struct timeval tv; gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
int main(int argc, char* argv[]) {
    // 解析参数
    int opt;
    while ((opt = getopt(argc, argv, "cghn:m:a:b:tvB:d")) != -1) {
        switch (opt) {
            case 'c': cpu_on = false; break;
            case 'g': gpu_on = false; break;
            case 'd': use_dp = true;  break;
            case 'n': n       = atoi(optarg); break;
            case 'm': samples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing  = true; break;
            case 'v': verbose = true; break;
            case 'B': blockSize = atoi(optarg); break;
            case 'h':
                puts("exponentialIntegral program");
                puts("usage: expint_exec [options]");
                puts("  -a value : interval start (default 0.0)");
                puts("  -b value : interval end   (default 10.0)");
                puts("  -c       : skip CPU");
                puts("  -g       : skip GPU");
                puts("  -d       : use Dynamic Parallelism (float only)");
                puts("  -n N     : highest order   (default 10)");
                puts("  -m M     : samples per order (default 10)");
                puts("  -B value : block size for CUDA kernel (default 128)");
                puts("  -t       : timing");
                puts("  -v       : verbose (print tables)");
                puts("  -h       : this help");
                return 0;
            default: return 1;
        }
    }

    if (!cpu_on && !gpu_on) {
        fprintf(stderr, "Error: both -c and -g specified - nothing to do!\n");
        return 1;
    }
    if (a >= b) { puts("Error: a >= b!"); return 1; }
    if (n == 0 || samples == 0) { puts("Error: n or samples = 0!"); return 1; }

    const double dx = (b - a) / double(samples);
    vector<vector<float>>  cpuFloat (n + 1, vector<float>(samples, 0.f));
    vector<vector<double>> cpuDouble(n + 1, vector<double>(samples, 0.0));
    double cpuTime = 0.0;

    // ---------- CPU ----------
    if (cpu_on) {
        double t0 = nowSeconds();
        for (unsigned int order = 0; order <= n; ++order) {
            for (unsigned int j = 0; j < samples; ++j) {
                double x = a + (j + 1) * dx;
                cpuFloat [order][j] = exponentialIntegralFloat (order, float (x));
                cpuDouble[order][j] = exponentialIntegralDouble(order,        x );
            }
        }
        cpuTime = nowSeconds() - t0;
    }

    if (timing && cpu_on)
        printf("CPU total time: %.6f s\n", cpuTime);
    if (verbose && cpu_on) {
        for (unsigned int order = 0; order <= n; ++order)
            for (unsigned int j = 0; j < samples; ++j) {
                double x = a + (j + 1) * dx;
                printf("CPU==> n=%2u x=%g  double=%-12.8g  float=%-12.8g\n",
                       order, x, cpuDouble[order][j], cpuFloat[order][j]);
            }
    }

#ifndef COMPILE_CPU_ONLY
    // ---------- GPU ----------
    vector<float> gpuFloat_flat((n + 1) * samples, 0.f);
    double gpu_total_time = 0.0;

    if (gpu_on) {
        vector<float> hx(samples);
        for (unsigned int j = 0; j < samples; ++j)
            hx[j] = float(a + (j + 1) * dx);

        double t0 = nowSeconds();

        if (use_dp) {
            gpu::expint_gpu_dp_float(n, hx.data(), gpuFloat_flat.data(), samples);
        } else {
            float* dx_d = nullptr; float* dy_d = nullptr;
            gpu::alloc_and_copy_to_device(hx.data(), dx_d, samples);
            cudaMalloc((void**)&dy_d, (n + 1) * samples * sizeof(float));
            gpu::expint_gpu_multi_float(n, dx_d, dy_d, samples, blockSize);
            cudaMemcpy(gpuFloat_flat.data(), dy_d, (n + 1) * samples * sizeof(float), cudaMemcpyDeviceToHost);
            gpu::free_device(dx_d); gpu::free_device(dy_d);
        }

        gpu_total_time = nowSeconds() - t0;
    }

    // ---------- 重组为二维结构 ----------
    vector<vector<float>> gpuFloat(n + 1, vector<float>(samples));
    for (unsigned int order = 0; order <= n; ++order)
        for (unsigned int j = 0; j < samples; ++j)
            gpuFloat[order][j] = gpuFloat_flat[j * (n + 1) + order];

    // ---------- 打印 GPU 输出 ----------
    if (verbose && gpu_on) {
        for (unsigned int order = 0; order <= n; ++order)
            for (unsigned int j = 0; j < samples; ++j) {
                double x = a + (j + 1) * dx;
                printf("GPU==> n=%2u x=%g  float=%.9g\n", order, x, gpuFloat[order][j]);
            }
    }

    // ---------- 计时输出 ----------
    if (timing && gpu_on) {
        printf("GPU total time: %.6f s\n", gpu_total_time);
        if (cpu_on)
            printf("Speed-up (CPU/GPU): %.2fx\n", cpuTime / gpu_total_time);
    }

    // ---------- 精度检查 ----------
    if (cpu_on && gpu_on) {
        int bad = 0;
        for (unsigned int order = 0; order <= n; ++order)
            for (unsigned int j = 0; j < samples; ++j)
                if (fabs(gpuFloat[order][j] - cpuFloat[order][j]) > 1e-5f)
                    ++bad;
        printf("[Precision Check] GPU vs CPU comparison: %s (threshold = 1e-5)\n",
               (bad == 0) ? "PASS" : "FAIL");
    }
#endif

    return 0;
}
