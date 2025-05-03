#include "radiator_gpu.h"
#include "cuda_helper.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// ---------------- Propagation Kernel with Shared Memory ----------------
__global__ void propagate_kernel(const float* in, float* out, int n, int m) {
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    int local_row = ty;
    int local_col = tx + 2;

    if (row < n && col < m)
        tile[local_row * (blockDim.x + 4) + local_col] = in[row * m + col];

    if (tx < 2 && row < n) {
        int halo_col = (col - 2 + m) % m;
        tile[local_row * (blockDim.x + 4) + tx] = in[row * m + halo_col];
    }

    if (tx >= blockDim.x - 2 && row < n) {
        int halo_col = (col + (tx - (blockDim.x - 2)) + 1) % m;
        tile[local_row * (blockDim.x + 4) + local_col + (tx - (blockDim.x - 2)) + 1] =
            in[row * m + halo_col];
    }

    __syncthreads();

    if (row >= n || col >= m) return;

    if (col == 0) {
        out[row * m + col] = in[row * m + col];
        return;
    }

    float sum = 1.60f * tile[local_row * (blockDim.x + 4) + (local_col - 2)]
              + 1.55f * tile[local_row * (blockDim.x + 4) + (local_col - 1)]
              + 1.00f * tile[local_row * (blockDim.x + 4) + (local_col)]
              + 0.60f * tile[local_row * (blockDim.x + 4) + (local_col + 1)]
              + 0.25f * tile[local_row * (blockDim.x + 4) + (local_col + 2)];

    out[row * m + col] = sum / 5.0f;
}

// ---------------- Warp Shuffle Row Average Kernel ----------------
__global__ void average_kernel(const float* matrix, float* averages, int n, int m) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int j = tid; j < m; j += blockDim.x)
        sum += matrix[row * m + j];

    // warp shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (tid % warpSize == 0) atomicAdd(&averages[row], sum);

    __syncthreads();

    if (tid == 0)
        averages[row] /= m;
}

// ---------------- GPU API ----------------
extern "C"
void gpu_propagate(float* d_in, float* d_out, int n, int m,
                   int block_x, int block_y, cudaStream_t stream) {
    dim3 threads(block_x, block_y);
    dim3 blocks((m + block_x - 1) / block_x, (n + block_y - 1) / block_y);
    size_t shared_mem = (block_x + 4) * block_y * sizeof(float);
    propagate_kernel<<<blocks, threads, shared_mem, stream>>>(d_in, d_out, n, m);
    CHECK_CUDA(cudaGetLastError());
}

extern "C"
void gpu_calculate_averages(float* d_matrix, float* d_avg, int n, int m,
                            int block_size, cudaStream_t stream) {
    dim3 blocks(n);
    dim3 threads(block_size);
    average_kernel<<<blocks, threads, 0, stream>>>(d_matrix, d_avg, n, m);
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

extern "C"
void gpu_alloc_averages(float** d_avg, int n) {
    CHECK_CUDA(cudaMalloc(d_avg, n * sizeof(float)));
}

extern "C"
void gpu_free_averages(float* d_avg) {
    if (d_avg) cudaFree(d_avg);
}

extern "C"
void validate_block_size(int block_x, int block_y) {
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

// ---------------- Validation ----------------
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
        printf("[Validation] Averages max diff = %.6e, mismatches > 1e-4 = %d\n",
               max_avg_diff, avg_mismatch);
    }
}
