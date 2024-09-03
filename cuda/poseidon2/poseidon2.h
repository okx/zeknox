#ifndef _POSEIDON2_H_
#define _POSEIDON2_H_

#include <stdint.h>
#include "types/int_types.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void cpu_poseidon2_hash_one(u64 *input, u32 size, u64 *data);
EXTERNC void cpu_poseidon2_hash_two(u64 *hash1, u64 *hash2, u64 *hash);

#ifdef USE_CUDA

#include "types/gl64_t.cuh"

EXTERNC __device__ void gpu_poseidon2_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash);
EXTERNC __device__ void gpu_poseidon2_hash_one_stride(gl64_t *inputs, u32 num_inputs, gl64_t *hash, u32 stride);
EXTERNC __device__ void gpu_poseidon2_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);
#endif

#endif // _POSEIDON2_H_
