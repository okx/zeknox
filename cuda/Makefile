$(shell ./configure.sh)
$(shell sleep 1)
include CudaArch.mk

# lib name
LIBNAME := "libcryptocuda"

# compilers
CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

# flags
INCL := -I.

# for release
CFLAGS := $(INCL) -Wall -Werror -O3 -march=native -fopenmp -fPIC
CXXFLAGS := $(INCL) -Wall -Wno-error=sign-compare -std=c++11 -O3 -march=native -fopenmp -fPIC
NVCCFLAGS := $(INCL) -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 -arch=$(CUDA_ARCH)

# for debug
# CFLAGS := $(INCL) -Wall -Werror -march=native -fopenmp -fPIC -g
# CXXFLAGS := $(INCL) -Wall -Wno-error=sign-compare -std=c++11 -march=native -fopenmp -fPIC -g
# NVCCFLAGS := $(INCL) -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -g -arch=$(CUDA_ARCH) -g

all: lib

lib: $(LIBNAME).a

clean:
	rm -f *.o *.so *.a *.exe
	rm -rf build1 build2

MKDIR_P ?= mkdir -p

BUILD_DIR_CXX := ./build1
BUILD_DIR_CUDA := ./build2

SRC_DIRS := ./merkle ./keccak ./poseidon ./monolith
SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.cu)
OBJS_CXX := $(SRCS:%=$(BUILD_DIR_CXX)/%.o)
OBJS_CUDA := $(SRCS:%=$(BUILD_DIR_CUDA)/%.o)

# c++ source
$(BUILD_DIR_CXX)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_CXX)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR_CXX)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -dc -o $@

$(BUILD_DIR_CUDA)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_CUDA)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -DUSE_CUDA $(NVCCFLAGS) $< -dc -o $@

$(LIBNAME).so: $(BUILD_DIR_CXX)/merkle/merkle.c.o $(BUILD_DIR_CUDA)/merkle/merkle.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon.cpp.o $(BUILD_DIR_CXX)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon.cu.o $(BUILD_DIR_CXX)/poseidon/element_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/element_bn128.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CXX)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CUDA)/poseidon2/poseidon2.cu.o
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $^

$(LIBNAME).a: $(BUILD_DIR_CXX)/merkle/merkle.c.o $(BUILD_DIR_CUDA)/merkle/merkle.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon.cpp.o $(BUILD_DIR_CXX)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon.cu.o $(BUILD_DIR_CXX)/poseidon/element_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/element_bn128.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CXX)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CUDA)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CXX)/monolith/monolith.cu.o $(BUILD_DIR_CUDA)/monolith/monolith.cu.o
	$(NVCC) $(NVCCFLAGS) -dlink -o $(BUILD_DIR_CUDA)/all_gpu.o $^
	ar cr $@ $(BUILD_DIR_CUDA)/all_gpu.o $^
	ranlib $@

tests.exe: $(BUILD_DIR_CUDA)/tests/tests.cu.o $(BUILD_DIR_CXX)/merkle/merkle.c.o $(BUILD_DIR_CUDA)/merkle/merkle.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon.cpp.o $(BUILD_DIR_CXX)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon.cu.o $(BUILD_DIR_CXX)/poseidon/element_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/element_bn128.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CXX)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CUDA)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CXX)/monolith/monolith.cu.o $(BUILD_DIR_CUDA)/monolith/monolith.cu.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lgomp -lgtest
