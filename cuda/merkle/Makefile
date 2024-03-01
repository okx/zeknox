$(shell ./configure.sh)
$(shell sleep 1)
include CudaArch.mk

CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

# for release
CFLAGS := -I../types -I../poseidon -I../keccak -Wall -Werror -O3 -march=native
CXXFLAGS := -I../types -I../poseidon -I../keccak -I. -Wall -Wno-error=sign-compare -std=c++11 -O3 -march=native
NVCCFLAGS := -I../types -I../poseidon -I../keccak -I../util -I.

# for debug
#CFLAGS := -I../types -I../poseidon -I../keccak -Wall -Werror -g
#CXXFLAGS := -I../types -I../poseidon -I../keccak -I. -Wall -Wno-error=sign-compare -std=c++11 -g
#NVCCFLAGS := -I../types -I../poseidon -I../keccak -I../util -I. -g

all: libgpu

lib: libgpu

libcpu: libmerkle.so

libgpu: libmerkle-gpu.so

clean:
	rm -f *.o *.so *.exe

libmerkle.so: merkle.h merkle.c ../poseidon/goldilocks.hpp ../poseidon/poseidon.hpp ../poseidon/poseidon.h ../poseidon/poseidon_bn128.cu ../poseidon/poseidon.cu ../keccak/keccak.cu
	$(CC) $(CFLAGS) -c -fPIC merkle.c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak.o
	$(CXX) -shared -o $@ poseidon.o poseidon_bn128.o keccak.o merkle.o -fopenmp

libmerkle-gpu.so: merkle.h merkle.c ../poseidon/goldilocks.hpp ../poseidon/poseidon.hpp ../poseidon/poseidon.cu merkle.cu
	$(CC) $(CFLAGS) -c -fPIC merkle.c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak_gpu.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file element.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file element_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128_gpu.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC merkle.cu -arch=$(CUDA_ARCH) -dc --output-file merkle_gpu.o
	$(NVCC) -Xcompiler -fopenmp -shared -arch=$(CUDA_ARCH) -o $@ merkle_gpu.o merkle.o poseidon.o poseidon_gpu.o element.o element_gpu.o poseidon_bn128.o poseidon_bn128_gpu.o keccak.o keccak_gpu.o

libmerkle-gpu.a: merkle.h merkle.c ../poseidon/goldilocks.hpp ../poseidon/poseidon.hpp ../poseidon/poseidon.cu merkle.cu
	$(CC) $(CFLAGS) -c -fPIC merkle.c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file keccak.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file keccak_gpu.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file poseidon_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file element.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file element_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file poseidon_bn128.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file poseidon_bn128_gpu.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC merkle.cu -arch=$(CUDA_ARCH) -dc -rdc=true --output-file merkle_gpu.o
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -dlink -o all_gpu.o merkle_gpu.o poseidon_gpu.o element_gpu.o poseidon_bn128_gpu.o keccak_gpu.o element.o poseidon_bn128.o keccak.o -lcudadevrt -lcudart
	ar cr $@ all_gpu.o merkle.o poseidon.o merkle_gpu.o poseidon_gpu.o element_gpu.o poseidon_bn128_gpu.o keccak_gpu.o element.o poseidon_bn128.o keccak.o
	ranlib libmerkle-gpu.a

test-rust.exe: merkle.c ../poseidon/libposeidon.a
	$(CC) $(CFLAGS) -g -DRUST_POSEIDON -DDEBUG $^ -o $@ -fopenmp

testcpu.exe: merkle.c
	$(CC) $(CFLAGS) -DTESTING -fPIC merkle.c -c -fopenmp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak.o
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file element.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ merkle.o keccak.o poseidon.o element.o poseidon_bn128.o

testgpu.exe: merkle.cu merkle.c ../poseidon/poseidon.cuh
	$(CC) $(CFLAGS) -c -fPIC merkle.c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../keccak/keccak.cu -arch=$(CUDA_ARCH) -dc --output-file keccak_gpu.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file element.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/element_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file element_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128.o
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC ../poseidon/poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_bn128_gpu.o
	$(NVCC) $(NVCCFLAGS) -DTESTING -DUSE_CUDA -Xcompiler -fopenmp -Xcompiler -fPIC merkle.cu -arch=$(CUDA_ARCH) -dc --output-file merkle_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ merkle_gpu.o merkle.o poseidon.o poseidon_gpu.o element.o element_gpu.o poseidon_bn128.o poseidon_bn128_gpu.o keccak.o keccak_gpu.o
