#ifndef _POSEIDON_H_
#define _POSEIDON_H_

#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void compute_hash_leaf(u64* digest, u64* data, u32 data_size);
EXTERNC void compute_hash_of_two(u64* digest, u64* digest_left, u64* digest_right);

#ifdef RUST_POSEIDON
/*
 * This is for testing only!
 *
 * These functions are implemented in Rust and imported as libposeidon.a 
*/
EXTERNC void ext_hash_or_noop(uint64_t *digest, uint64_t *data, uint64_t data_count);
EXTERNC void ext_hash_of_two(uint64_t *digest, uint64_t *digest_left, uint64_t *digest_right);
#endif // RUST_POSEIDON

#endif // _POSEIDON_H_