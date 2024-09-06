#ifndef __MERKLE_H__
#define __MERKLE_H__

#include <stdint.h>
#include "types/int_types.h"

#define HASH_SIZE 32

#define HASH_SIZE_U64 4

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

enum HashType
{
    HashPoseidon = 0,
    HashKeccak = 1,
    HashPoseidonBN128 = 2,
    HashPoseidon2 = 3,
    HashMonolith = 4
};

EXTERNC void fill_digests_buf_linear_gpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type,
    u64 gpu_id);

EXTERNC void fill_digests_buf_linear_multigpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type);

EXTERNC void fill_digests_buf_linear_cpu(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    const void *leaves_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type);

#ifdef __USE_AVX__
EXTERNC void fill_digests_buf_linear_cpu_avx(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    const void *leaves_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type);
#endif // __USE_AVX__

#ifdef __AVX512__
EXTERNC void fill_digests_buf_linear_cpu_avx512(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    const void *leaves_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type);
#endif // __AVX512__

#endif // __MERKEL_H__
