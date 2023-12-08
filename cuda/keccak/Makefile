CC := gcc
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

all: test_keccak

test_keccak_v1: test_keccak.cpp keccak_sha3.cu ../poseidon/libposeidon.a
	$(NVCC) -I../util -I../merkle keccak_sha3.cu -dc
	$(CXX) -I../merkle test_keccak.cpp -c
	$(NVCC) test_keccak.o keccak_sha3.o ../poseidon/libposeidon.a -o $@ -L/usr/local/cuda/lib64 -lcudart

test_keccak: test_keccak.c keccak-tiny.c ../poseidon/libposeidon.a
	$(CC) -I../util -I../merkle $^ -o $@
