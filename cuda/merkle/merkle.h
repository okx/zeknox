#ifndef __MERKLE_H__
#define __MERKLE_H__

#include "types.h"

#include "poseidon.h"

#define HASH_SIZE   32

#define HASH_SIZE_U64   4

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void fill_digests_buf_in_c(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
);

EXTERNC void fill_digests_buf_in_rounds_in_c(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
);

EXTERNC void fill_digests_buf_in_rounds_in_c_on_gpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
);

EXTERNC void fill_init(u64 digests_count, u64 leaves_count, u64 caps_count, u64 leaf_size, u64 hash_size);

EXTERNC void fill_init_rounds(u64 leaves_count, u64 rounds);

EXTERNC void fill_delete();

EXTERNC void fill_delete_rounds();

EXTERNC u64* get_digests_ptr();

EXTERNC u64* get_cap_ptr();

EXTERNC u64* get_leaves_ptr();

#endif // __MERKEL_H__
