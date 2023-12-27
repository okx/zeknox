#ifndef _POSEIDON_BN128_H_
#define _POSEIDON_BN128_H_

#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void poseidon_bn128_hash_leaf(u64* digest, u64* data, u32 data_size);
EXTERNC void poseidon_bn128_hash_of_two(u64* digest, u64* digest_left, u64* digest_right);

#ifdef USE_CUDA
#include "gl64_t.cuh"

EXTERNC __device__ void gpu_poseidon_bn128_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash);
EXTERNC __device__ void gpu_poseidon_bn128_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);
#endif

#endif // _POSEIDON_BN128_H_
