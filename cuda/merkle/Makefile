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
INCL := -I../types -I../poseidon -I../poseidon2 -I../monolith -I../keccak -I../util

# for release
CFLAGS := $(INCL) -Wall -Werror -O3 -march=native -fopenmp -fPIC
CXXFLAGS := $(INCL) -Wall -Wno-error=sign-compare -std=c++17 -O3 -march=native -fopenmp -fPIC
NVCCFLAGS := $(INCL) -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 -arch=$(CUDA_ARCH)

# for debug
#CFLAGS := $(INCL) -Wall -Werror -march=native -fopenmp -fPIC -g
#CXXFLAGS := $(INCL) -Wall -Wno-error=sign-compare -std=c++17 -march=native -fopenmp -fPIC -g
#NVCCFLAGS := $(INCL) -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -g -arch=$(CUDA_ARCH) -g

clean:
	rm -f *.o *.so *.a *.exe
	rm -rf build1 build2

MKDIR_P ?= mkdir -p

BUILD_DIR_CXX := ./build1
BUILD_DIR_CUDA := ./build2

SRC_DIRS := ./ ../keccak ../poseidon ../monolith
SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.cu | cut -d '/' -f 2-)
OBJS_CXX := $(SRCS:%=$(BUILD_DIR_CXX)/%.o)
OBJS_CUDA := $(SRCS:%=$(BUILD_DIR_CUDA)/%.o)

# c++ source
$(BUILD_DIR_CXX)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_CXX)/%.c.o: ../%.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR_CXX)/%.cu.o: ../%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -dc -o $@

$(BUILD_DIR_CUDA)/%.cu.o: ../%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -DUSE_CUDA $(NVCCFLAGS) $< -dc -o $@

testcpu.exe: merkle.cpp
	$(CXX) $(CXXFLAGS) -DTESTING -fPIC merkle.cpp -c -fopenmp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak.o
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon2/poseidon2.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon2.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file element.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../monolith/monolith.cu -arch=$(CUDA_ARCH) -dc --output-file monolith.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ merkle.o keccak.o poseidon.o element.o poseidon_bn128.o monolith.o poseidon2.o

testcpuavx512.exe: merkle.cpp
	$(CXX) $(CXXFLAGS) -DTESTING -fPIC -D__USE_AVX__ -D__AVX512__ -D__NO_GMP__ -mavx2 -mavx512f ../poseidon/poseidon.cpp merkle.cpp -o $@

testgpu.exe: merkle.cpp merkle_c.h merkle.cu merkle.h $(BUILD_DIR_CXX)/poseidon/poseidon.cpp.o $(BUILD_DIR_CXX)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon.cu.o $(BUILD_DIR_CXX)/poseidon/element_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/element_bn128.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CXX)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CUDA)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CXX)/monolith/monolith.cu.o $(BUILD_DIR_CUDA)/monolith/monolith.cu.o
	$(CXX) $(CXXFLAGS) -c merkle.cpp -o $(BUILD_DIR_CXX)/merkle.o
	$(NVCC) $(NVCCFLAGS) -DTESTING -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC merkle.cu -arch=$(CUDA_ARCH) -dc --output-file $(BUILD_DIR_CUDA)/merkle_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $(BUILD_DIR_CXX)/poseidon/poseidon.cpp.o $(BUILD_DIR_CXX)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/keccak/keccak.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon.cu.o $(BUILD_DIR_CXX)/poseidon/element_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/element_bn128.cu.o $(BUILD_DIR_CXX)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CUDA)/poseidon/poseidon_bn128.cu.o $(BUILD_DIR_CXX)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CUDA)/poseidon2/poseidon2.cu.o $(BUILD_DIR_CXX)/monolith/monolith.cu.o $(BUILD_DIR_CUDA)/monolith/monolith.cu.o $(BUILD_DIR_CUDA)/merkle_gpu.o $(BUILD_DIR_CXX)/merkle.o