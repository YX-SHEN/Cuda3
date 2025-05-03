#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <getopt.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "cuda_helper.h"
#include "radiator_cpu.h"
#include "radiator_gpu.h"
#include "timer.h"
#include "timer_cuda.h"

// 参数
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
    printf("  -t             Print timing and speedup\n");
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

    if (n <= 0 || m <= 2 || p <= 0 || block_x <= 0 || block_y <= 0) {
        fprintf(stderr, "Invalid input parameters.\n");
        exit(1);
    }

    if (n % block_y != 0 || m % block_x != 0) {
        fprintf(stderr, "Error: Matrix size must be divisible by block size.\n");
        exit(1);
    }

    validate_block_size(block_x, block_y);
}

void initialize_single_matrix(float* A, int n, int m) {
    for (int i = 0; i < n; ++i) {
        float base = 0.98f * (i + 1) * (i + 1) / (n * n);
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

    float* initial_matrix = new float[n * m];
    float* cpu_result = nullptr;
    float* gpu_result = new float[n * m];
    float* cpu_avg = nullptr;
    float* gpu_avg = nullptr;

    if (!gpu_only) {
        float* A = new float[n * m];
        float* B = new float[n * m];
        initialize_matrices(A, B, n, m);
        std::copy(A, A + n * m, initial_matrix);
        delete[] A; delete[] B;
    } else {
        initialize_single_matrix(initial_matrix, n, m);
    }

    // ======== CPU 计算 ========
    double cpu_time = 0.0;
    if (!gpu_only) {
        cpu_result = new float[n * m];
        float* A = new float[n * m];
        float* B = new float[n * m];
        std::copy(initial_matrix, initial_matrix + n * m, A);
        std::copy(initial_matrix, initial_matrix + n * m, B);

        Timer cpu_timer;
        cpu_timer.start();
        for (int i = 0; i < p; ++i) {
            propagate_heat(A, B, n, m);
            std::swap(A, B);
        }
        cpu_time = cpu_timer.elapsed();
        std::copy(A, A + n * m, cpu_result);

        if (calculate_average) {
            cpu_avg = new float[n];
            calculate_row_averages(A, cpu_avg, n, m);
        }

        delete[] A;
        delete[] B;
    }

    // ======== GPU 计算 ========
    float *d_in = nullptr, *d_out = nullptr, *d_avg = nullptr;
    float alloc_time = 0.0f, h2d_time = 0.0f, compute_time = 0.0f, avg_time = 0.0f, d2h_time = 0.0f;

    try {
        CUDATimer timer;

        timer.start();
        gpu_alloc_memory(&d_in, &d_out, n, m);
        if (calculate_average)
            CHECK_CUDA(cudaMalloc(&d_avg, n * sizeof(float)));
        timer.stop();
        alloc_time = timer.elapsed();

        timer.start();
        copy_to_device(d_in, initial_matrix, n, m);
        timer.stop();
        h2d_time = timer.elapsed();

        timer.start();
        for (int i = 0; i < p; ++i) {
            gpu_propagate(d_in, d_out, n, m, block_x, block_y, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            std::swap(d_in, d_out);
        }
        timer.stop();
        compute_time = timer.elapsed();

        if (calculate_average) {
            timer.start();
            gpu_calculate_averages(d_in, d_avg, n, m, block_x, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            timer.stop();
            avg_time = timer.elapsed();
        }

        timer.start();
        copy_from_device(gpu_result, d_in, n, m);
        if (calculate_average) {
            gpu_avg = new float[n];
            copy_averages_from_device(gpu_avg, d_avg, n);
        }
        timer.stop();
        d2h_time = timer.elapsed();

        if (timing) {
            printf("\n[GPU Timing Results]\n");
            printf("Alloc Time    : %.3f ms\n", alloc_time);
            printf("H2D Transfer  : %.3f ms\n", h2d_time);
            printf("Compute Time  : %.3f ms\n", compute_time);
            if (calculate_average)
                printf("Row Avg Time  : %.3f ms\n", avg_time);
            printf("D2H Transfer  : %.3f ms\n", d2h_time);
            float total_gpu = alloc_time + h2d_time + compute_time + avg_time + d2h_time;
            printf("Total GPU Time: %.3f ms\n", total_gpu);
            if (!gpu_only)
                printf("Speedup (CPU/GPU Compute): %.2fx\n", (cpu_time * 1000.0) / compute_time);
        }

        if (!gpu_only) {
            validate_results(cpu_result, gpu_result, cpu_avg, gpu_avg, n, m, calculate_average);
        }

    } catch (std::exception& e) {
        fprintf(stderr, "CUDA Exception: %s\n", e.what());
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (d_avg) cudaFree(d_avg);
    }

    delete[] initial_matrix;
    delete[] gpu_result;
    if (cpu_result) delete[] cpu_result;
    if (cpu_avg) delete[] cpu_avg;
    if (gpu_avg) delete[] gpu_avg;

    return 0;
}
