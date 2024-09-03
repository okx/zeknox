#ifndef _POSEIDON_BN128_H_
#define _POSEIDON_BN128_H_

#include <stdint.h>
#include "types/int_types.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void cpu_poseidon_bn128_hash_one(u64 *input, u32 size, u64 *hash);
EXTERNC void cpu_poseidon_bn128_hash_two(u64 *hash1, u64 *hash2, u64 *hash);

#ifdef USE_CUDA

#include "types/gl64_t.cuh"

EXTERNC __device__ void gpu_poseidon_bn128_hash_one(gl64_t *input, u32 size, gl64_t *hash);
EXTERNC __device__ void gpu_poseidon_bn128_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);
#endif

#endif // _POSEIDON_BN128_H_
