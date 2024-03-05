CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH = sm_86

# for debug
CFLAGS := -I../types -I../poseidon -I../keccak -Wall -Werror -g
CXXFLAGS := -I../types -I../util -Wall -Wno-error=sign-compare -std=c++11 -g
NVCCFLAGS := -I../types -I../poseidon -I../keccak -I../util -I. -g

all: test_poseidon2_goldilocks_gpu.exe

clean:
	rm -f *.o *.so *.exe

test_poseidon2_goldilocks_gpu.exe: poseidon2.cuh poseidon2.cu poseidon2.h test_poseidon2.cu
	$(NVCC) -DUSE_CUDA -DDEBUG -Xcompiler -fPIC $(NVCCFLAGS) poseidon2.cu -arch=$(CUDA_ARCH) -dc -o poseidon2_gpu.o
	$(NVCC) -DUSE_CUDA -DDEBUG -Xcompiler -fPIC $(NVCCFLAGS) ../poseidon/poseidon.cu -arch=$(CUDA_ARCH) -dc -o poseidon_gpu.o
	$(NVCC) -DDEBUG -Xcompiler -fPIC $(NVCCFLAGS) poseidon2.cu -arch=$(CUDA_ARCH) -dc -o poseidon2.o
	$(NVCC) -DUSE_CUDA -Xcompiler -fPIC $(NVCCFLAGS) test_poseidon2.cu -arch=$(CUDA_ARCH) -dc -o test_poseidon2_gpu.o
	$(NVCC) -Xcompiler -fPIC $(NVCCFLAGS) -arch=$(CUDA_ARCH) poseidon2.o test_poseidon2_gpu.o poseidon2_gpu.o poseidon_gpu.o -o $@ -L/usr/local/cuda/
