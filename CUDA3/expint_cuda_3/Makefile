# Makefile for expint_cuda project with DP support

NVCC      = nvcc
CXX       = g++
INCLUDES  = -Iinclude
CFLAGS    = -O2 -std=c++11 -Wall
NVFLAGS   = -O2 -std=c++11 -arch=sm_70 -rdc=true
LDFLAGS   = -lcudadevrt

SRC       = main.cpp src/expint_gpu.cu src/expint_gpu_dp.cu
TARGET    = bin/expint_exec

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf bin/expint_exec \
	       *.o src/*.o \
	       logs/*.txt \
	       *.ptx *.sass *.cubin *.fatbin \
	       *.cu.cudafe* *.linkinfo *.mod.c \
	       *.dSYM core.*

.PHONY: all clean
