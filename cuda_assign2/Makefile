# Makefile for CUDA + C++ heat radiator project

# 编译器设置
CXX      := nvcc
CXXFLAGS := -O2 -std=c++17 -Xcompiler="-fopenmp"
INCLUDES := -I./src
ARCH     := -arch=sm_75

# 文件路径
SRCDIR   := src
OBJDIR   := obj
TARGET   := radiator_exec

# 源文件
SRCS := $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/*.cu)
OBJS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(filter %.cpp,$(SRCS))) \
        $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(filter %.cu,$(SRCS)))

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCH) $(INCLUDES) -o $@ $^

# 生成中间目标文件
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(ARCH) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(CXX) $(CXXFLAGS) $(ARCH) $(INCLUDES) -c $< -o $@

# 清理目标
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)
