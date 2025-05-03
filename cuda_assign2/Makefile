# Makefile - Task 1 (CPU-only build)

CXX := g++
CXXFLAGS := -O2 -std=c++11 -Wall
SRC_DIR := src
BIN := radiator_exec

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/radiator_cpu.cpp
OBJS := $(SRCS:.cpp=.o)

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(BIN)

.PHONY: all clean
