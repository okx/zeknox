#ifndef __MERKLE_PRIVATE_H__
#define __MERKLE_PRIVATE_H__

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

typedef struct {
    int left_index, right_index, target_index;
} HashTask;

extern u64* global_digests_buf;
extern u64* global_leaves_buf;
extern u64* global_cap_buf;
extern u64* global_digests_buf_end;
extern u64* global_leaves_buf_end;
extern u64* global_cap_buf_end;

// pointers to the actual hash functions (could be Poseidon or Keccak)
extern void (*cpu_hash_one_ptr)(u64* input, u32 size, u64* hash);
extern void (*cpu_hash_two_ptr)(u64 *hash1, u64 *hash2, u64 *hash);

extern int* leaf_index;
extern HashTask* internal_index;
extern int* round_size;
extern int max_round;
extern int max_round_size;

EXTERNC void fill_subtree_get_index(
    int target_index,
    int digests_offset,
    int digests_count,
    int leaves_offset,
    int leaves_count,
    int leaf_size,
    int round);

EXTERNC void init_gpu_functions(u64 hash_type);

#ifdef DEBUG

#include <stdio.h>

void print_hash(u64* hash) {
    for (int i = 0; i < 4; i++) {
        printf("%lu ", hash[i]);
    }
    printf("\n");
}

#endif

#endif  // __MERKLE_PRIVATE_H__
