# compilers
CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

CUDA_ARCH=sm_86

# for release
CFLAGS := $(INCL) -O3 -march=native -fopenmp -fPIC -DSECP256K1_WIDEMUL_INT64
NVCCFLAGS := $(INCL) -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 -arch=$(CUDA_ARCH) -DSECP256K1_WIDEMUL_INT64 -DSECP256K1_CPLUSPLUS_TEST_OVERRIDE

MKDIR_P ?= mkdir -p

BUILD_DIR_CXX := ./build1
BUILD_DIR_CUDA := ./build2

SRC_DIRS := ./
SRCS_CXX := $(shell find $(SRC_DIRS) -name "*.c")
SRCS_CUDA := $(shell find $(SRC_DIRS) -name "*.cu")
OBJS_CXX := $(SRCS_CXX:%=$(BUILD_DIR_CXX)/%.o)
OBJS_CUDA := $(SRCS_CUDA:%=$(BUILD_DIR_CUDA)/%.o)

$(BUILD_DIR_CXX)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR_CUDA)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -DUSE_CUDA $(NVCCFLAGS) $< -dc -o $@

test_cpu.exe: $(OBJS_CXX)
	$(CC) $(CFLAGS) $(OBJS_CXX) -o $@

test_gpu.exe: $(OBJS_CUDA)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f *.o *.so *.a *.exe
	rm -rf build1 build2