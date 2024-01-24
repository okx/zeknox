CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH = sm_86

# for debug
CFLAGS := -I../types -I../poseidon -I../keccak -Wall -Werror -g
CXXFLAGS := -I../types -I../util -Wall -Wno-error=sign-compare -std=c++11 -g
NVCCFLAGS := -I../types -I../poseidon -I../keccak -I../util -I. -g

clean:
	rm -f *.o *.so *.exe

test_poseidon_bn128_cpu.exe: poseidon_bn128.cu element_bn128.cu
	$(NVCC) -Xcompiler -fPIC $(NVCCFLAGS) -DTESTING element_bn128.cu -arch=$(CUDA_ARCH) -dc
	$(NVCC) -Xcompiler -fPIC $(NVCCFLAGS) -DTESTING poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc
	$(NVCC) -Xcompiler -fPIC $(NVCCFLAGS) -DTESTING -arch=$(CUDA_ARCH) element_bn128.o poseidon_bn128.o -o $@ -L/usr/local/cuda/lib64 -lcudart

test_poseidon_bn128_gpu.exe: poseidon_bn128.cu element_bn128.cu
	$(NVCC) -Xcompiler -fPIC -DUSE_CUDA $(NVCCFLAGS) -DTESTING element_bn128.cu -arch=$(CUDA_ARCH) -dc
	$(NVCC) -Xcompiler -fPIC -DUSE_CUDA $(NVCCFLAGS) -DTESTING poseidon_bn128.cu -arch=$(CUDA_ARCH) -dc
	$(NVCC) -Xcompiler -fPIC -DUSE_CUDA $(NVCCFLAGS) -DTESTING -arch=$(CUDA_ARCH) element_bn128.o poseidon_bn128.o -o $@ -L/usr/local/cuda/lib64 -lcudart