#ifndef __MERKLE_C_H__
#define __MERKLE_C_H__

#include "int_types.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void fill_digests_buf_linear_cpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type);

#endif // __MERKLE_C_H__