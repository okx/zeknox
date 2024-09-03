#ifndef _POSEIDON_H_
#define _POSEIDON_H_

#include <stdint.h>
#include "types/int_types.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void cpu_poseidon_hash_one(u64 *input, u32 size, u64 *data);
EXTERNC void cpu_poseidon_hash_two(u64 *hash1, u64 *hash2, u64 *hash);

#ifdef USE_CUDA

#include "types/gl64_t.cuh"

EXTERNC __device__ void gpu_poseidon_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash);
EXTERNC __device__ void gpu_poseidon_hash_one_stride(gl64_t *inputs, u32 num_inputs, gl64_t *hash, u32 stride);
EXTERNC __device__ void gpu_poseidon_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);
#endif

#ifdef RUST_POSEIDON
/*
 * This is for testing only!
 *
 * These functions are implemented in Rust and imported as libposeidon.a
*/
EXTERNC void ext_poseidon_hash_or_noop(u64 *digest, u64 *data, u64 data_count);
EXTERNC void ext_poseidon_hash_of_two(u64 *digest, u64 *digest_left, u64 *digest_right);
#endif // RUST_POSEIDON

#endif // _POSEIDON_H_
