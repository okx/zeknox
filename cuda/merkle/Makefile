CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH = sm_86

CFLAGS := -I../poseidon -Wall -Werror -O3
CXXFLAGS := -I../poseidon -I. -Wall -Wno-error=sign-compare -std=c++11 -O3 -march=native
NVCCFLAGS := -I../poseidon -I.

all: libmerkle.so

lib: libmerkle.so

libgpu: libmerkle-gpu.so

clean:
	rm -f *.o *.so

libmerkle.so: merkle.h merkle.c ../poseidon/goldilocks.hpp ../poseidon/poseidon.hpp ../poseidon/poseidon.h
	$(CC) $(CFLAGS) -c -fPIC merkle.c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(CXX) -shared -o $@ poseidon.o merkle.o -fopenmp	

libmerkle-gpu.so: merkle.h merkle.c ../poseidon/goldilocks.hpp ../poseidon/poseidon.hpp merkle.cu
	$(CC) $(CFLAGS) -c -fPIC merkle.c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC merkle.cu -arch=$(CUDA_ARCH) -dc --output-file merkle_gpu.o
	$(NVCC) -Xcompiler -fopenmp -shared -arch=$(CUDA_ARCH) -o $@ merkle_gpu.o merkle.o poseidon.o

test-rust: merkle.c ../poseidon/libposeidon.a
	$(CC) $(CFLAGS) -g -DRUST_POSEIDON -DDEBUG $^ -o $@ -fopenmp

test-cpp: merkle.c ../poseidon/goldilocks.hpp ../poseidon/poseidon.hpp ../poseidon/poseidon.cpp
	$(CC) -I../poseidon/ -g -DDEBUG $^ -o $@ -fopenmp
#	$(CC) -I../poseidon/ -DDEBUG -O3 -march=native $^ -o $@

testgpu: merkle.cu merkle.c ../poseidon/poseidon.cuh
	$(CC) $(CFLAGS) merkle.c -c -fopenmp
	$(CXX) $(CXXFLAGS) -c -fPIC ../poseidon/poseidon.cpp	
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -Xcompiler -fPIC -DTESTING -DDEBUG merkle.cu -arch=$(CUDA_ARCH) -dc --output-file merkle_gpu.o
	$(NVCC) -Xcompiler -fopenmp merkle_gpu.o merkle.o poseidon.o -arch=$(CUDA_ARCH) -g -o $@ -L/usr/local/cuda/lib64 -lcudart -lgmp
