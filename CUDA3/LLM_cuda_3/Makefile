# Makefile for MAP55616-03 - CUDA Exponential Integral Calculation

CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++11
NVCCFLAGS = -O2 -std=c++11

all: llm_expint_exec

llm_expint_exec: main.cpp expint_cuda.cu
	$(NVCC) $(NVCCFLAGS) main.cpp expint_cuda.cu -o llm_expint_exec

clean:
	rm -f llm_expint_exec
