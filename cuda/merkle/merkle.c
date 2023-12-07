#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

// TODO - remove
#include <stdio.h>

#include "types.h"
#include "merkle.h"
#include "merkle_private.h"
#include "poseidon.h"

// Global vars
u64* global_digests_buf = NULL;
u64* global_leaves_buf = NULL;
u64* global_cap_buf = NULL;
u64* global_digests_buf_end = NULL;
u64* global_leaves_buf_end = NULL;
u64* global_cap_buf_end = NULL;

void (*cpu_hash_one_ptr)(u64* digest, u64* data, u32 data_size);
void (*cpu_hash_two_ptr)(u64* digest, u64* digest_left, u64* digest_right);

int* leaf_index = NULL;
HashTask* internal_index = NULL;
int* round_size = NULL;
int max_round = 0;
int max_round_size = 0;

/**
 * Version 1 - Fill in a recursive way (the same as in Plonky2 Rust code)
 */
void fill_subtree(u64* target_hash, u64* digests, u64 digests_count, u64* leaves, u64 leaves_count, u64 leaf_size, char target_type) {
    if (target_type == 0) {
        assert(target_hash >= global_cap_buf && target_hash < global_cap_buf_end);
    }
    else {
        assert(target_hash >= global_digests_buf && target_hash < global_digests_buf_end);
    }
    assert(digests >= global_digests_buf);
    assert(leaves >= global_leaves_buf);
    assert(digests + digests_count < global_digests_buf_end);
    assert(leaves + leaves_count < global_leaves_buf_end);    

    if (digests_count == 0) {
#ifdef RUST_POSEIDON
        ext_hash_or_noop(target_hash, leaves, leaf_size);
#else
        cpu_hash_one_ptr(target_hash, leaves, leaf_size);
#endif
    } else {
        u64 mid = digests_count / 2;
        u64* left_digests = digests;
        u64* left_digest = digests + (mid-1) * HASH_SIZE_U64;
        u64* right_digest = digests + mid * HASH_SIZE_U64;
        u64* right_digests = digests + (mid+1) * HASH_SIZE_U64;
        u64 new_digests_count = mid - 1;
        mid = leaves_count / 2;
        u64* left_leaves = leaves;
        u64* right_leaves = leaves + mid * leaf_size;
        u64 new_leaves_count = mid;
        fill_subtree(left_digest, left_digests, new_digests_count, left_leaves, new_leaves_count, leaf_size, 1);
        fill_subtree(right_digest, right_digests, new_digests_count, right_leaves, new_leaves_count, leaf_size, 1);
#ifdef RUST_POSEIDON
        ext_hash_of_two(target_hash, left_digest, right_digest);
#else
        cpu_hash_two_ptr(target_hash, left_digest, right_digest);
#endif
    }    
}

void fill_digests_buf_in_c(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
) {
    if (cap_buf_size == leaves_buf_size) {
        u64* cptr = global_cap_buf;
        u64* lptr = global_leaves_buf;

        for (u64 i = 0; i < leaves_buf_size; i++) {
            assert(cptr < global_cap_buf_end);
            assert(lptr < global_leaves_buf_end);
#ifdef RUST_POSEIDON
            ext_hash_or_noop(cptr, lptr, leaf_size);
#else
            cpu_hash_one_ptr(cptr, lptr, leaf_size);
#endif
            cptr += HASH_SIZE_U64;
            lptr += leaf_size;            
        }
        return;	
    }

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    u64* cptr = global_cap_buf;
    u64* lptr = global_leaves_buf;
    u64* dptr = global_digests_buf;
    for (u64 k = 0; k < cap_buf_size; k++) {
        assert(cptr < global_cap_buf_end);
        assert(lptr < global_leaves_buf_end);
        assert(dptr < global_digests_buf_end);
        fill_subtree(
            cptr, 
            dptr, 
            subtree_digests_len, 
            lptr,
            subtree_leaves_len,
            leaf_size, 0);
        cptr += HASH_SIZE_U64;
        dptr += HASH_SIZE_U64 * subtree_digests_len;
        lptr += leaf_size * subtree_leaves_len;
    }
}

/**
 * Version 2 - Iterative algorithm
 */
void fill_subtree_get_index(int target_index, int digests_offset, int digests_count, int leaves_offset, int leaves_count, int leaf_size, int round) {
    if (digests_count == 0) {
        leaf_index[leaves_offset] = target_index;
    }
    else {
        int mid = digests_count / 2;
        int left_digests_offset = digests_offset;
        int left_digest_index = digests_offset + (mid-1);
        int right_digest_index = digests_offset + mid;
        int right_digests_offset = digests_offset + (mid+1);
        int new_digests_count = mid - 1;
        mid = leaves_count / 2;
        int left_leaves_offset = leaves_offset;
        int right_leaves_offset = leaves_offset + mid;
        int new_leaves_count = mid;
        fill_subtree_get_index(left_digest_index, left_digests_offset, new_digests_count, left_leaves_offset, new_leaves_count, leaf_size, round+1);
        fill_subtree_get_index(right_digest_index, right_digests_offset, new_digests_count, right_leaves_offset, new_leaves_count, leaf_size, round+1);
        HashTask* ht = &internal_index[round * max_round_size + round_size[round]];
        ht->target_index = target_index;
        ht->left_index = left_digest_index;
        ht->right_index = right_digest_index;
        round_size[round]++;
        if (round > max_round) {
            max_round = round;
        }
    }
}

void fill_subtree_in_rounds(int leaves_count, int leaf_size) {
    // leaves
#pragma omp parallel for
    for (int i = 0; i < leaves_count; i++) {
#ifdef RUST_POSEIDON
        ext_hash_or_noop(global_digests_buf + (leaf_index[i] * HASH_SIZE_U64), global_leaves_buf + (i * leaf_size), leaf_size);
#else
        cpu_hash_one_ptr(global_digests_buf + (leaf_index[i] * HASH_SIZE_U64), global_leaves_buf + (i * leaf_size), leaf_size);
#endif
    }
    // internal rounds on digest buffer
    for (int r = max_round; r > 0; r--) {
#pragma omp parallel for        
        for (int i = 0; i < round_size[r]; i++) {
            HashTask* ht = &internal_index[r * max_round_size + i];
#ifdef RUST_POSEIDON
            ext_hash_of_two(global_digests_buf + (ht->target_index * HASH_SIZE_U64), global_digests_buf + (ht->left_index * HASH_SIZE_U64), global_digests_buf + (ht->right_index * HASH_SIZE_U64));
#else
            cpu_hash_two_ptr(global_digests_buf + (ht->target_index * HASH_SIZE_U64), global_digests_buf + (ht->left_index * HASH_SIZE_U64), global_digests_buf + (ht->right_index * HASH_SIZE_U64));
#endif
        }
    }
    // cap buffer
// #pragma omp parallel for
    for (int i = 0; i < round_size[0]; i++) {
        HashTask* ht = &internal_index[i];
#ifdef RUST_POSEIDON
        ext_hash_of_two(global_cap_buf + (ht->target_index * HASH_SIZE_U64), global_digests_buf + (ht->left_index * HASH_SIZE_U64), global_digests_buf + (ht->right_index * HASH_SIZE_U64));
#else
        cpu_hash_two_ptr(global_cap_buf + (ht->target_index * HASH_SIZE_U64), global_digests_buf + (ht->left_index * HASH_SIZE_U64), global_digests_buf + (ht->right_index * HASH_SIZE_U64));
#endif

    }
}

void fill_digests_buf_in_rounds_in_c(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height
) {
    if (cap_buf_size == leaves_buf_size) {
#pragma omp parallel for
        for (u64 i = 0; i < leaves_buf_size; i++) {
            u64* cptr = global_cap_buf + (i * HASH_SIZE_U64);
            u64* lptr = global_leaves_buf + (i * leaf_size);
            assert(cptr < global_cap_buf_end);
            assert(lptr < global_leaves_buf_end);
#ifdef RUST_POSEIDON
            ext_hash_or_noop(cptr, lptr, leaf_size);
#else
            cpu_hash_one_ptr(cptr, lptr, leaf_size);
#endif
        }        
        return;	
    }

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);
    
    for (u64 k = 0; k < cap_buf_size; k++) {
        fill_subtree_get_index(k, k * subtree_digests_len, subtree_digests_len, k * subtree_leaves_len, subtree_leaves_len, leaf_size, 0); 
    }
    fill_subtree_in_rounds(leaves_buf_size, leaf_size);
}

void fill_digests_buf_linear_cpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height)
{    
    if (cap_buf_size == leaves_buf_size)
    {
#pragma omp parallel for
        for (int i = 0; i < leaves_buf_size; i++) {
#ifdef RUST_POSEIDON
            ext_hash_or_noop(global_digests_buf + i * HASH_SIZE_U64, global_leaves_buf + (i * leaf_size), leaf_size);
#else
            cpu_hash_one_ptr(global_digests_buf + i * HASH_SIZE_U64, global_leaves_buf + (i * leaf_size), leaf_size);
#endif
        }
        return;
    }   

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);
    
    // for all the subtrees
    for (u32 k = 0; k < cap_buf_size; k++)
    {        
        u64 *leaves_buf_ptr = global_leaves_buf + k * subtree_leaves_len * leaf_size;
        u64 *digests_buf_ptr = global_digests_buf + k * subtree_digests_len * HASH_SIZE_U64;
        u64 *cap_buf_ptr = global_cap_buf + k * HASH_SIZE_U64;

        // if one leaf => return it hash
        if (subtree_leaves_len == 1)
        {
            cpu_hash_one_ptr(digests_buf_ptr, leaves_buf_ptr, leaf_size);
            memcpy(cap_buf_ptr, digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            cpu_hash_one_ptr(digests_buf_ptr, leaves_buf_ptr, leaf_size);
            cpu_hash_one_ptr(digests_buf_ptr + HASH_SIZE_U64, leaves_buf_ptr + leaf_size, leaf_size);
            cpu_hash_two_ptr(cap_buf_ptr, digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64);
            continue;
        }

        // 2. compute leaf hashes
        u64 *digests_curr_ptr = digests_buf_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;   
#pragma omp parallel for        
        for (u32 i = 0; i < subtree_leaves_len; i++) {
#ifdef RUST_POSEIDON
            ext_hash_or_noop(digests_curr_ptr + (i * HASH_SIZE_U64), leaves_buf_ptr + (i * leaf_size), leaf_size);
#else
            cpu_hash_one_ptr(digests_curr_ptr + (i * HASH_SIZE_U64), leaves_buf_ptr + (i * leaf_size), leaf_size);
#endif
        }    

        // 3. compute internal hashes
        u32 r = (u32)log2(subtree_leaves_len) - 1;
        u32 last_index = subtree_digests_len - subtree_leaves_len;
        
        for (; r > 0; r--)
        {
            last_index -= (1 << r);
            // printf("CPU round %d Last idx %d\n", r, last_index);
            u64 *digests_buf_ptr2 = digests_buf_ptr + last_index * HASH_SIZE_U64;

#pragma omp parallel for             
            for (int idx = 0; idx < (1 << r); idx++)
            {
                u32 left_idx = 2 * (idx + 1) + last_index;
                u32 right_idx = left_idx + 1;
                u64 *left_ptr = digests_buf_ptr2 + (left_idx * HASH_SIZE_U64);
                u64 *right_ptr = digests_buf_ptr2 + (right_idx * HASH_SIZE_U64);
                // printf("%lu %lu\n", *left_ptr, *right_ptr);
#ifdef RUST_POSEIDON
                ext_hash_of_two(digests_buf_ptr2 + (idx * HASH_SIZE_U64), left_ptr, right_ptr);
#else
                cpu_hash_two_ptr(digests_buf_ptr2 + (idx * HASH_SIZE_U64), left_ptr, right_ptr);
#endif
            }            
        }

        // 4. compute cap hashes
#ifdef RUST_POSEIDON
        ext_hash_of_two(cap_buf_ptr, digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64);
#else        
        cpu_hash_two_ptr(cap_buf_ptr, digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64);
#endif

    } // end for k    
}

/**
 * Common auxiliary functions.
 */
void fill_init(u64 digests_count, u64 leaves_count, u64 caps_count, u64 leaf_size, u64 hash_size, u64 hash_type) {   
    u64 mem_size_digests_buf = ((((digests_count + 1) * HASH_SIZE_U64 * sizeof(u64)) >> 12) + 1) << 12;
    u64 mem_size_leaves_buf = ((((leaves_count + 1) * leaf_size * sizeof(u64)) >> 12) + 1) << 12;
    u64 mem_size_cap_buf = ((((caps_count + 1) * HASH_SIZE_U64 * sizeof(u64)) >> 12) + 1) << 12;

    global_digests_buf = (u64*)malloc(mem_size_digests_buf);
    global_leaves_buf = (u64*)malloc(mem_size_leaves_buf);
    global_cap_buf = (u64*)malloc(mem_size_cap_buf);
    assert(global_digests_buf != NULL);
    assert(global_leaves_buf != NULL);
    assert(global_cap_buf != NULL);
    global_digests_buf_end = global_digests_buf + ((digests_count + 1) * HASH_SIZE_U64);
    global_leaves_buf_end = global_leaves_buf + ((leaves_count + 1) * leaf_size);
    global_cap_buf_end = global_cap_buf + ((caps_count + 1) * HASH_SIZE_U64);

    if (hash_type == 0) {
        cpu_hash_one_ptr = &poseidon_hash_leaf;
        cpu_hash_two_ptr = &poseidon_hash_of_two;
    }    
    init_gpu_functions(hash_type);
}

void fill_delete() {
    free(global_digests_buf);
    free(global_cap_buf);
    free(global_leaves_buf);
    global_digests_buf = NULL;
    global_cap_buf = NULL;
    global_leaves_buf = NULL;
}

void fill_init_rounds(u64 leaves_count, u64 rounds) {
    max_round_size = leaves_count / 2;
    leaf_index = (int*)malloc(leaves_count * sizeof(int));
    internal_index = (HashTask*)malloc(rounds * max_round_size * sizeof(HashTask));
    memset(internal_index, 0, rounds * max_round_size * sizeof(HashTask));
    round_size = (int*)malloc(rounds * sizeof(int));
    memset(round_size, 0, rounds * sizeof(int));
    max_round = 0;
}

void fill_delete_rounds() {
    free(internal_index);
    free(leaf_index);
    free(round_size);
}

u64* get_digests_ptr() {
    return global_digests_buf;
}

u64* get_cap_ptr() {
    return global_cap_buf;
}

u64* get_leaves_ptr() {
    return global_leaves_buf;
}

/**
 * End of library code.
 */

// #define DEBUG
#ifdef DEBUG

#define LEAF_SIZE_U64 1

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
// #include "test_leaves.h"

void generate_random_leaves(u64* leaves, u32 n_leaves, u32 leaf_size) {
    srand(time(NULL));
    for (int i = 0; i < n_leaves * leaf_size; i++) {
        u32 r = rand();
        leaves[i] = ((u64)r << 32) + r * 88958514;
    }
}

int main() {    
    u32 cap_h = 3;
    u64 n_caps = (1 << cap_h);
    u64 n_leaves = 1024;
    u64 n_digests = 2 * (n_leaves - n_caps);
    // u64 rounds = 10;

    /*                                                                      
    digests_buf = (u64*)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    cap_buf = (u64*)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    // leaves = (u64*)test_leaves_1024;
    cap_buf_end = cap_buf + n_caps * HASH_SIZE_U64;
    digests_buf_end = digests_buf + n_digests * HASH_SIZE_U64;
    leaves_end = leaves + n_leaves * LEAF_SIZE_U64;
    */

    printf("Digests %lu, Leaves %lu, Caps %lu, Leaf size %u, \n", n_digests, n_leaves, n_caps, LEAF_SIZE_U64);
    fill_init(n_digests, n_leaves, n_caps, LEAF_SIZE_U64, HASH_SIZE_U64);
    generate_random_leaves(global_leaves_buf, n_leaves, LEAF_SIZE_U64);

    // fill_init_rounds(n_leaves, rounds, LEAF_SIZE_U64);

    // fill_digests_buf_in_rounds_in_c(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, cap_h);

    fill_digests_buf_in_c(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, cap_h);

/*
    printf("After fill...\n");
    for (int i = 0; i < n_digests; i++)
        print_hash(digests_buf + i * HASH_SIZE_U64);
    printf("Cap...\n");
    for (int i = 0; i < n_caps; i++)
        print_hash(cap_buf + i * HASH_SIZE_U64);
*/
    // fill_delete_rounds(rounds);

    fill_delete();

    // free(digests_buf);
    // free(cap_buf);

    return 0;
}

// This is to compare the outputs of our 2 versions
int main2() {    
    u64 n_caps = 1024;
    u64 n_leaves = 1024;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = 10;
                                                                      
    global_digests_buf = (u64*)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf = (u64*)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    // leaves = (u64*)test_leaves_1024;
    global_cap_buf_end = global_cap_buf + n_caps * HASH_SIZE_U64;
    global_digests_buf_end = global_digests_buf + n_digests * HASH_SIZE_U64;
    global_leaves_buf_end = global_leaves_buf + n_leaves * 7;

    fill_init_rounds(n_leaves, rounds);

    fill_digests_buf_in_rounds_in_c(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, 1);

    u64* digests_buf2 = (u64*)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    memcpy(digests_buf2, global_digests_buf, n_digests * HASH_SIZE_U64 * sizeof(u64));

    fill_digests_buf_in_c(n_digests, n_caps, n_leaves, 7, 1);

    int is_diff = 0;
    u64* ptr1 = global_digests_buf;
    u64* ptr2 = digests_buf2;
    for (int i = 0; i < n_digests * HASH_SIZE_U64; i++, ptr1++, ptr2++) {
        if (*ptr1 != *ptr2) {
            is_diff = 1;
            break;
        }
    }
    if (is_diff) {
        printf("Test failed: outputs are different!\n");
    }
    else {
        printf("Test passed: outputs are the same!\n");
    }

    fill_delete_rounds(rounds);

    free(global_digests_buf);
    free(digests_buf2);
    free(global_cap_buf);

    return 0;
}
#endif  // DEBUG
