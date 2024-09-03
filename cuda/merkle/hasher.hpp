#ifndef _HASHER_CUH_
#define _HASHER_CUH_

#include <stdint.h>

#ifdef USE_CUDA
#include "types/gl64_t.cuh"
#endif

class Hasher {
public:

#ifdef USE_CUDA
__host__ __device__ Hasher() {};
__host__ virtual void cpu_hash_one(uint64_t *input, uint64_t size, uint64_t *output) = 0;
__host__ virtual void cpu_hash_two(uint64_t *input1, uint64_t *input2, uint64_t *output) = 0;
__device__ virtual void gpu_hash_one(gl64_t *input, uint32_t size, gl64_t *output) = 0;
__device__ virtual void gpu_hash_two(gl64_t *input1, gl64_t *input2, gl64_t *output) = 0;
__host__ __device__ ~Hasher() {};
#else
virtual void cpu_hash_one(uint64_t *input, uint64_t size, uint64_t *output) = 0;
virtual void cpu_hash_two(uint64_t *input1, uint64_t *input2, uint64_t *output) = 0;
virtual ~Hasher() {};
#endif

};

#endif // _HASHER_CUH_