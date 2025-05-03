#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <algorithm>
#include <cmath>
#include "radiator_cpu.h"
#include "radiator_gpu.h"
#include "timer.h"
#include "timer_cuda.h"

int n = 32, m = 32, p = 10;
int block_x = 16, block_y = 16;
bool calculate_average = false;
bool gpu_only = false;
bool timing = false;

void print_usage() {
    printf("Usage: ./radiator_exec [options]\n");
    printf("Options:\n");
    printf("  -n <rows>      Number of rows (default 32)\n");
    printf("  -m <cols>      Number of cols (default 32)\n");
    printf("  -p <steps>     Iterations (default 10)\n");
    printf("  -x <block_x>   GPU block x (default 16)\n");
    printf("  -y <block_y>   GPU block y (default 16)\n");
    printf("  -a             Calculate row averages\n");
    printf("  -c             GPU only (skip CPU)\n");
    printf("  -t             Print timing\n");
    printf("  -h             Help\n");
}

void parse_arguments(int argc, char** argv) {
    int opt;
    while ((opt = getopt(argc, argv, "n:m:p:x:y:acth")) != -1) {
        switch (opt) {
            case 'n': n = atoi(optarg); break;
            case 'm': m = atoi(optarg); break;
            case 'p': p = atoi(optarg); break;
            case 'x': block_x = atoi(optarg); break;
            case 'y': block_y = atoi(optarg); break;
            case 'a': calculate_average = true; break;
            case 'c': gpu_only = true; break;
            case 't': timing = true; break;
            case 'h': print_usage(); exit(0);
            default: print_usage(); exit(1);
        }
    }

    if (n <= 0 || m <= 0 || p <= 0 || block_x <= 0 || block_y <= 0) {
        fprintf(stderr, "Invalid input parameters.\n");
        exit(1);
    }

    if (n % block_y != 0 || m % block_x != 0) {
        fprintf(stderr, "Error: Matrix size must be divisible by block size.\n");
        exit(1);
    }
}

// 用于 GPU-only 模式下生成一份单矩阵初始化
void initialize_single_matrix(float* A, int n, int m) {
    for (int i = 0; i < n; ++i) {
        const float base = 0.98f * (i + 1) * (i + 1) / (n * n);
        A[i * m] = base;
        for (int j = 1; j < m; ++j) {
            float ratio = float((m - j) * (m - j)) / (m * m);
            A[i * m + j] = base * ratio;
        }
    }
}

int main(int argc, char** argv) {
    parse_arguments(argc, argv);
    printf("Matrix: %dx%d, Iterations: %d, Block: %dx%d\n", n, m, p, block_x, block_y);

    float* matrix_initial = new float[n * m];
    float* gpu_result = new float[n * m];

    if (!gpu_only) {
        // CPU 初始化（不执行实际传播）
        float* A = new float[n * m];
        float* B = new float[n * m];
        initialize_matrices(A, B, n, m);
        std::copy(A, A + n * m, matrix_initial);
        delete[] A;
        delete[] B;
    } else {
        // GPU-only 模式下初始化
        initialize_single_matrix(matrix_initial, n, m);
    }

    // GPU 执行路径（空内核结构）
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDATimer timer;
    float alloc_time, h2d_time, compute_time, d2h_time;

    try {
        timer.start();
        gpu_alloc_memory(&d_in, &d_out, n, m);
        alloc_time = timer.elapsed();

        timer.start();
        copy_to_device(d_in, matrix_initial, n, m);
        h2d_time = timer.elapsed();

        timer.start();
        for (int step = 0; step < p; ++step) {
            gpu_propagate(d_in, d_out, n, m, block_x, block_y, 0); // 空壳
            std::swap(d_in, d_out);
        }
        compute_time = timer.elapsed();

        timer.start();
        copy_from_device(gpu_result, d_in, n, m);
        d2h_time = timer.elapsed();

        gpu_free_memory(d_in, d_out);

        if (timing) {
            printf("[GPU] Alloc: %.2f ms, H2D: %.2f ms, Compute: %.2f ms, D2H: %.2f ms\n",
                   alloc_time, h2d_time, compute_time, d2h_time);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "GPU Error: %s\n", e.what());
    }

    delete[] matrix_initial;
    delete[] gpu_result;
    return 0;
}
