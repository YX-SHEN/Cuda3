#ifndef TIMER_CUDA_H
#define TIMER_CUDA_H

#include <cuda_runtime.h>

class CUDATimer {
public:
    CUDATimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }

    float elapsed() {
        float ms;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

#endif // TIMER_CUDA_H
