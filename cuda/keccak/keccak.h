#ifndef _KECCAK_H_
#define _KECCAK_H_

#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void cpu_keccak_hash_one(uint64_t *data, uint32_t data_size, uint64_t *digest);
EXTERNC void cpu_keccak_hash_two(uint64_t *digest_left, uint64_t *digest_right, uint64_t *digest);

#ifdef USE_CUDA

#include "gl64_t.cuh"

EXTERNC __device__ void gpu_keccak_hash_one(gl64_t *input, uint32_t size, gl64_t *hash);
EXTERNC __device__ void gpu_keccak_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);

#endif

#ifdef RUST_POSEIDON
/*
 * This is for testing only!
 *
 * These functions are implemented in Rust and imported as libposeidon.a
 */
EXTERNC void ext_keccak_hash_or_noop(uint64_t *digest, uint64_t *data, uint64_t data_count);
EXTERNC void ext_keccak_hash_of_two(uint64_t *digest, uint64_t *digest_left, uint64_t *digest_right);
#endif // RUST_POSEIDON

#endif // _POSEIDON_H_