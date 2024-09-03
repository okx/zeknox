#ifndef _HASHER_CUH_
#define _HASHER_CUH_

#include <stdint.h>

#ifdef USE_CUDA
#include "types/gl64_t.cuh"
#endif

#include "types/int_types.h"

class Hasher {
public:

#ifdef USE_CUDA
__host__ __device__ Hasher() {};
__host__ virtual void cpu_hash_one(u64 *input, u64 size, u64 *output) = 0;
__host__ virtual void cpu_hash_two(u64 *input1, u64 *input2, u64 *output) = 0;
__device__ virtual void gpu_hash_one(gl64_t *input, u32 size, gl64_t *output) = 0;
__device__ virtual void gpu_hash_two(gl64_t *input1, gl64_t *input2, gl64_t *output) = 0;
__host__ __device__ ~Hasher() {};
#else
virtual void cpu_hash_one(u64 *input, u64 size, u64 *output) = 0;
virtual void cpu_hash_two(u64 *input1, u64 *input2, u64 *output) = 0;
virtual ~Hasher() {};
#endif

};

#endif // _HASHER_CUH_