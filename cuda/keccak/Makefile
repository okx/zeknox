CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH := sm_86

all: test_keccak_cpu.exe

test_keccak_cpu.exe: keccak.cu
	$(NVCC) -I../util -I../types keccak.cu -dc
	$(CC) -I../types test_keccak.c -c
	$(NVCC) test_keccak.o keccak.o -o $@ -L/usr/local/cuda/lib64 -lcudart

test_keccak_gpu.exe: keccak.cu
	$(NVCC) -I../util -I../types keccak.cu -arch=$(CUDA_ARCH) -dc -o keccak.o
	$(NVCC) -DUSE_CUDA -I../util -I../types keccak.cu -arch=$(CUDA_ARCH) -dc -o keccak_gpu.o
	$(CC) -I../types test_keccak.c -c
	$(NVCC) test_keccak.o keccak.o keccak_gpu.o -arch=$(CUDA_ARCH) -o $@ -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -rf *.o *.exe
