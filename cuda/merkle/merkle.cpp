#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "int_types.h"
#include "merkle.h"
#include "merkle_private.h"
#include "merkle_c.h"

#include "hasher.hpp"
#include "keccak.hpp"
#include "poseidon.hpp"
#include "poseidon_bn128.hpp"
#include "poseidon2.hpp"
#include "monolith.hpp"

// Global vars
u64 *global_digests_buf = NULL;
u64 *global_leaves_buf = NULL;
u64 *global_cap_buf = NULL;
u64 *global_digests_buf_end = NULL;
u64 *global_leaves_buf_end = NULL;
u64 *global_cap_buf_end = NULL;

int *global_leaf_index = NULL;
HashTask *global_internal_index = NULL;
int *global_round_size = NULL;
int global_max_round = 0;
int global_max_round_size = 0;

void fill_digests_buf_linear_cpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type)
{
    Hasher *hasher = new PoseidonHasher();
    switch (hash_type) {
        case HashKeccak:
            hasher = new KeccakHasher();
            break;
        case HashPoseidon2:
            hasher = new Poseidon2Hasher();
            break;
        case HashPoseidonBN128:
            hasher = new PoseidonBN128Hasher();
            break;
        case HashMonolith:
            hasher = new MonolithHasher();
            break;
        default:
            break;
    }

    if (cap_buf_size == leaves_buf_size)
    {
#pragma omp parallel for
        for (u64 i = 0; i < leaves_buf_size; i++)
        {
#ifdef RUST_POSEIDON
            ext_poseidon_hash_or_noop(global_digests_buf + i * HASH_SIZE_U64, global_leaves_buf + (i * leaf_size), leaf_size);
#else
            hasher->cpu_hash_one(global_leaves_buf + (i * leaf_size), leaf_size, global_digests_buf + i * HASH_SIZE_U64);
#endif
        }
        delete hasher;
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
            hasher->cpu_hash_one(leaves_buf_ptr, leaf_size, digests_buf_ptr);
            memcpy(cap_buf_ptr, digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            hasher->cpu_hash_one(leaves_buf_ptr, leaf_size, digests_buf_ptr);
            hasher->cpu_hash_one(leaves_buf_ptr + leaf_size, leaf_size, digests_buf_ptr + HASH_SIZE_U64);
            hasher->cpu_hash_two(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);
            continue;
        }

        // 2. compute leaf hashes
        u64 *digests_curr_ptr = digests_buf_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;
#pragma omp parallel for
        for (u32 i = 0; i < subtree_leaves_len; i++)
        {
#ifdef RUST_POSEIDON
            ext_poseidon_hash_or_noop(digests_curr_ptr + (i * HASH_SIZE_U64), leaves_buf_ptr + (i * leaf_size), leaf_size);
#else
            hasher->cpu_hash_one(leaves_buf_ptr + (i * leaf_size), leaf_size, digests_curr_ptr + (i * HASH_SIZE_U64));
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
                ext_poseidon_hash_of_two(digests_buf_ptr2 + (idx * HASH_SIZE_U64), left_ptr, right_ptr);
#else
                hasher->cpu_hash_two(left_ptr, right_ptr, digests_buf_ptr2 + (idx * HASH_SIZE_U64));
#endif
            }
        }

        // 4. compute cap hashes
#ifdef RUST_POSEIDON
        ext_poseidon_hash_of_two(cap_buf_ptr, digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64);
#else
        hasher->cpu_hash_two(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);
#endif

    } // end for k

    delete hasher;
}

/**
 * End of library code.
 */

// #define TESTING
#ifdef TESTING

#define LEAF_SIZE_U64 8

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

void generate_random_leaves(u64 *leaves, u32 n_leaves, u32 leaf_size)
{
    srand(time(NULL));
    for (u64 i = 0; i < n_leaves * leaf_size; i++)
    {
        u32 r = rand();
        leaves[i] = ((u64)r << 32) + r * 88958514;
    }
}

int main()
{
    u64 cap_h = 1;
    u64 n_caps = (1 << cap_h);
    u64 n_leaves = 1024;
    u64 n_digests = 2 * (n_leaves - n_caps);

    global_leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));
    global_digests_buf = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf_end = global_cap_buf + n_caps * HASH_SIZE_U64;
    global_digests_buf_end = global_digests_buf + n_digests * HASH_SIZE_U64;
    global_leaves_buf_end = global_leaves_buf + n_leaves * 7;

    generate_random_leaves(global_leaves_buf, n_leaves, LEAF_SIZE_U64);

    fill_digests_buf_linear_cpu(
        n_digests,
        n_caps,
        n_leaves,
        LEAF_SIZE_U64,
        cap_h,
        HashPoseidon
    );

    free(global_digests_buf);
    free(global_leaves_buf);
    free(global_cap_buf);

    return 0;
}
#endif // TESTING
