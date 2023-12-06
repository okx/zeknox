#ifndef __MERKLE_H__
#define __MERKLE_H__

#include <stdint.h>

#define HASH_SIZE   32

#define HASH_SIZE_U64   4

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void fill_digests_buf_in_c(
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height
);

EXTERNC void fill_digests_buf_in_rounds_in_c(
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height
);

EXTERNC void fill_digests_buf_in_rounds_in_c_on_gpu(
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height
);

EXTERNC void fill_digests_buf_linear_cpu(
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height
);

EXTERNC void fill_digests_buf_linear_gpu(
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height
);

EXTERNC void fill_init(
    uint64_t digests_count, 
    uint64_t leaves_count, 
    uint64_t caps_count, 
    uint64_t leaf_size, 
    uint64_t hash_size
);

EXTERNC void fill_init_rounds(
    uint64_t leaves_count, 
    uint64_t rounds
);

EXTERNC void fill_delete();

EXTERNC void fill_delete_rounds();

EXTERNC uint64_t* get_digests_ptr();

EXTERNC uint64_t* get_cap_ptr();

EXTERNC uint64_t* get_leaves_ptr();

#endif // __MERKEL_H__
