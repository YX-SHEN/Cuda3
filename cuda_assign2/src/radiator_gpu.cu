#include "radiator_gpu.h"
#include "cuda_helper.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ---------------- Propagation Kernel ----------------
__global__ void propagate_kernel(const float* in, float* out, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= m) return;

    if (col == 0) {
        out[row * m + col] = in[row * m + col];
        return;
    }

    int jm2 = (col - 2 + m) % m;
    int jm1 = (col - 1 + m) % m;
    int jp1 = (col + 1) % m;
    int jp2 = (col + 2) % m;

    float sum = 1.60f * in[row * m + jm2]
              + 1.55f * in[row * m + jm1]
              + 1.00f * in[row * m + col]
              + 0.60f * in[row * m + jp1]
              + 0.25f * in[row * m + jp2];

    out[row * m + col] = sum / 5.0f;
}

// ---------------- Row Average Kernel ----------------
__global__ void average_kernel(const float* matrix, float* averages, int n, int m) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    for (int j = tid; j < m; j += stride) {
        sum += matrix[row * m + j];
    }
    sdata[tid] = sum;

    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        averages[row] = sdata[0] / m;
    }
}

// ---------------- GPU API ----------------
extern "C"
void gpu_propagate(float* d_in, float* d_out, int n, int m,
                   int block_x, int block_y, cudaStream_t stream) {
    dim3 threads(block_x, block_y);
    dim3 blocks((m + block_x - 1) / block_x, (n + block_y - 1) / block_y);
    propagate_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n, m);
    CHECK_CUDA(cudaGetLastError());
}

extern "C"
void gpu_calculate_averages(float* d_matrix, float* d_avg, int n, int m,
                            int block_size, cudaStream_t stream) {
    dim3 blocks(n);
    dim3 threads(block_size);
    size_t shared = block_size * sizeof(float);
    average_kernel<<<blocks, threads, shared, stream>>>(d_matrix, d_avg, n, m);
    CHECK_CUDA(cudaGetLastError());
}

// ---------------- Memory Management ----------------
extern "C"
void gpu_alloc_memory(float** d_in, float** d_out, int n, int m) {
    size_t bytes = n * m * sizeof(float);
    CHECK_CUDA(cudaMalloc(d_in, bytes));
    CHECK_CUDA(cudaMalloc(d_out, bytes));
}

extern "C"
void gpu_free_memory(float* d_in, float* d_out) {
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
}

extern "C" void validate_block_size(int block_x, int block_y) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (block_x <= 0 || block_y <= 0) {
        fprintf(stderr, "[ERROR] Block size must be positive: block_x = %d, block_y = %d\n",
                block_x, block_y);
        exit(EXIT_FAILURE);
    }

    if (block_x * block_y > prop.maxThreadsPerBlock) {
        fprintf(stderr, "[ERROR] Block size %d×%d exceeds max threads per block (%d)\n",
                block_x, block_y, prop.maxThreadsPerBlock);
        exit(EXIT_FAILURE);
    }
}

extern "C"
void gpu_alloc_averages(float** d_avg, int n) {
    CHECK_CUDA(cudaMalloc(d_avg, n * sizeof(float)));
}

extern "C"
void gpu_free_averages(float* d_avg) {
    if (d_avg) cudaFree(d_avg);
}

// ---------------- Memory Transfers ----------------
extern "C"
void copy_to_device(float* d_in, const float* h_in, int n, int m) {
    CHECK_CUDA(cudaMemcpy(d_in, h_in, n * m * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C"
void copy_from_device(float* h_out, const float* d_out, int n, int m) {
    CHECK_CUDA(cudaMemcpy(h_out, d_out, n * m * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C"
void copy_averages_from_device(float* h_avg, const float* d_avg, int n) {
    CHECK_CUDA(cudaMemcpy(h_avg, d_avg, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ---------------- Validation (CPU-only) ----------------
extern "C"
void validate_results(const float* cpu_matrix, const float* gpu_matrix,
                      const float* cpu_avg, const float* gpu_avg,
                      int n, int m, bool has_avg) {
    float max_diff = 0.0f;
    int mismatch = 0;

    for (int i = 0; i < n * m; ++i) {
        float diff = fabs(cpu_matrix[i] - gpu_matrix[i]);
        if (diff > 1e-4f) mismatch++;
        max_diff = std::max(max_diff, diff);
    }
    printf("[Validation] Matrix max diff = %.6e, mismatches > 1e-4 = %d\n", max_diff, mismatch);

    if (has_avg) {
        float max_avg_diff = 0.0f;
        int avg_mismatch = 0;
        for (int i = 0; i < n; ++i) {
            float diff = fabs(cpu_avg[i] - gpu_avg[i]);
            if (diff > 1e-4f) avg_mismatch++;
            max_avg_diff = std::max(max_avg_diff, diff);
        }
        printf("[Validation] Averages max diff = %.6e, mismatches > 1e-4 = %d\n", max_avg_diff, avg_mismatch);
    }
}
