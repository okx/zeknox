#ifndef __KECCAK_CUH__
#define __KECCAK_CUH__

#include "int_types.h"
#include "gl64_t.cuh"
#include "cuda_utils.cuh"

__device__ void gpu_keccak_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash);

__device__ void gpu_keccak_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);

#endif      // __KECCAK_CUH__