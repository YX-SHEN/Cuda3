# Makefile for CUDA + C++ hybrid project

# 编译器设置
NVCC        = nvcc
CXX         = g++
CXXFLAGS    = -O2 -std=c++17
NVCCFLAGS   = -O2 -std=c++17 -Xcompiler="-fopenmp" -arch=sm_75

# 路径
INCLUDE     = -I./src
SRC_DIR     = src
OBJ_DIR     = obj
BIN         = radiator_exec

# 源文件列表
CPP_SRCS    = $(SRC_DIR)/main.cpp $(SRC_DIR)/radiator_cpu.cpp
CU_SRCS     = $(SRC_DIR)/radiator_gpu.cu
OBJS        = $(OBJ_DIR)/main.o $(OBJ_DIR)/radiator_cpu.o $(OBJ_DIR)/radiator_gpu.o

# 生成目标
all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ $^

# 使用 nvcc 编译包含 cuda_runtime.h 的 .cpp 文件（如 main.cpp）
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/radiator_cpu.o: $(SRC_DIR)/radiator_cpu.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/radiator_gpu.o: $(SRC_DIR)/radiator_gpu.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

# 清理
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN)

# 自动创建 obj 目录（可选）
$(shell mkdir -p $(OBJ_DIR))
