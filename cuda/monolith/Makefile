CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH = sm_86

# for debug
CFLAGS := -I../types -I../poseidon -I../keccak -Wall -Werror -g
CXXFLAGS := -I../types -I../util -Wall -Wno-error=sign-compare -std=c++11 -g
NVCCFLAGS := -I../types -I../poseidon -I../keccak -I../util -I. -g

all: test_monolith_goldilocks_gpu.exe

clean:
	rm -f *.o *.so *.exe

test_monolith_goldilocks_gpu.exe: monolith.cuh monolith.cu monolith.h test_monolith.cu
	$(NVCC) -DUSE_CUDA -DDEBUG -Xcompiler -fPIC $(NVCCFLAGS) monolith.cu -arch=$(CUDA_ARCH) -dc -o monolith_gpu.o
	$(NVCC) -DUSE_CUDA -DDEBUG -Xcompiler -fPIC $(NVCCFLAGS) ../poseidon/poseidon.cu -arch=$(CUDA_ARCH) -dc -o poseidon_gpu.o
	$(NVCC) -DDEBUG -Xcompiler -fPIC $(NVCCFLAGS) monolith.cu -arch=$(CUDA_ARCH) -dc -o monolith.o
	$(NVCC) -DUSE_CUDA -Xcompiler -fPIC $(NVCCFLAGS) test_monolith.cu -arch=$(CUDA_ARCH) -dc -o test_monolith_gpu.o
	$(NVCC) -Xcompiler -fPIC $(NVCCFLAGS) -arch=$(CUDA_ARCH) monolith.o test_monolith_gpu.o monolith_gpu.o poseidon_gpu.o -o $@ -L/usr/local/cuda/
