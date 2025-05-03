# 编译器与编译选项
CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra -I./src
NVFLAGS  = -O3 -std=c++11 -I./src

# 文件路径
SRC_DIR  = src
CPPSRC   = $(SRC_DIR)/main.cpp $(SRC_DIR)/radiator_cpu.cpp
CU_SRC   = $(SRC_DIR)/radiator_gpu.cu
HEADERS  = $(SRC_DIR)/radiator_cpu.h $(SRC_DIR)/radiator_gpu.h $(SRC_DIR)/timer.h $(SRC_DIR)/timer_cuda.h

# 对象文件
OBJ_CPP  = $(CPPSRC:.cpp=.o)
OBJ_CU   = $(CU_SRC:.cu=.o)
TARGET   = radiator_exec

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	$(NVCC) $(NVFLAGS) -o $@ $^

# main.cpp 用 nvcc 编译（因为它引用了 cuda_runtime.h）
$(SRC_DIR)/main.o: $(SRC_DIR)/main.cpp $(HEADERS)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# 其他 .cpp 文件用 g++ 编译
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# .cu 文件编译
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)
