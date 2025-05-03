#ifndef RADIATOR_GPU_H
#define RADIATOR_GPU_H

#include <cuda_runtime.h>

// GPU API
extern "C" void validate_block_size(int block_x, int block_y);
extern "C" void gpu_alloc_memory(float** d_in, float** d_out, int n, int m);
extern "C" void gpu_free_memory(float* d_in, float* d_out);
extern "C" void copy_to_device(float* d_in, const float* h_in, int n, int m);
extern "C" void copy_from_device(float* h_out, const float* d_out, int n, int m);

extern "C" void gpu_propagate(float* d_in, float* d_out, int n, int m,
                              int block_x, int block_y, cudaStream_t stream);

extern "C" void gpu_calculate_averages(float* d_matrix, float* d_avg, int n, int m,
                                       int block_size, cudaStream_t stream);

extern "C" void gpu_alloc_averages(float** d_avg, int n);
extern "C" void gpu_free_averages(float* d_avg);
extern "C" void copy_averages_from_device(float* h_avg, const float* d_avg, int n);

extern "C" void validate_results(const float* cpu_matrix, const float* gpu_matrix,
                                 const float* cpu_avg, const float* gpu_avg,
                                 int n, int m, bool has_avg);

#endif  // RADIATOR_GPU_H
