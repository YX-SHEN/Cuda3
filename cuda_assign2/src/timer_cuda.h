#ifndef TIMER_CUDA_H
#define TIMER_CUDA_H

#include "cuda_helper.h"

class CUDATimer {
public:
    CUDATimer() : start_(nullptr), stop_(nullptr) {
        cudaError_t err;

        err = cudaEventCreate(&start_);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDATimer: Failed to create start_ event: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaEventCreate(&stop_);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDATimer: Failed to create stop_ event: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    ~CUDATimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_)  cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CHECK_CUDA(cudaEventRecord(stop_, stream));
    }

    float elapsed() {
        CHECK_CUDA(cudaEventSynchronize(stop_));
        float ms = -1.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

#endif // TIMER_CUDA_H
