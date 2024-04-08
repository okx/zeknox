CC := gcc
CXX := g++
HIPCC := hipcc

all: test_keccak_cpu.exe

test_keccak_cpu.exe: keccak_cpu.c test_keccak.c
	$(CC) $^ -o $@

test_keccak_gpu.exe: keccak_cpu.c test_keccak_hip.hip
	$(HIPCC) -I../util $^ -o $@

clean:
	rm -rf *.o *.exe
