#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Unified macro for all CUDA error checks
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                  \
do {                                                                      \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error (%s:%d): %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)
#endif

#endif // CUDA_HELPER_H
