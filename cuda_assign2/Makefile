# Compiler settings
NVCC       = nvcc
NVCCFLAGS  = -O2 -std=c++17 -arch=sm_75 -Xcompiler="-fopenmp"
INCLUDES   = -I./src

# Source files
SRC       = src/main.cpp src/radiator_cpu.cpp src/radiator_gpu.cu
OBJ_DIR   = obj
OBJS      = $(SRC:src/%.cpp=$(OBJ_DIR)/%.o)
OBJS      := $(OBJS:src/%.cu=$(OBJ_DIR)/%.o)

TARGET    = radiator_exec

# Default target
all: $(TARGET)

# Link target
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^

# Compile all source files with nvcc
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
