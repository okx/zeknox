#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "types/int_types.h"
#include "merkle/merkle.h"
#include "merkle/merkle_c.h"
#include "merkle/hasher.hpp"
#include "poseidon/poseidon.hpp"
#include "poseidon2/poseidon2.hpp"
#include "poseidon/poseidon_bn128.hpp"
#include "keccak/keccak.hpp"
#include "monolith/monolith.hpp"

void fill_digests_buf_linear_cpu(
    u64 *digests_buf_ptr,
    u64 *cap_buf_ptr,
    u64 *leaves_buf_ptr,
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
            ext_poseidon_hash_or_noop(cap_buf_ptr + (i * HASH_SIZE_U64), leaves_buf_ptr + (i * leaf_size), leaf_size);
#else
            hasher->cpu_hash_one(leaves_buf_ptr + (i * leaf_size), leaf_size, cap_buf_ptr + (i * HASH_SIZE_U64));
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
        u64 *curr_leaves_buf_ptr = leaves_buf_ptr + k * subtree_leaves_len * leaf_size;
        u64 *curr_digests_buf_ptr = digests_buf_ptr + k * subtree_digests_len * HASH_SIZE_U64;
        u64 *curr_cap_buf_ptr = cap_buf_ptr + k * HASH_SIZE_U64;

        // if one leaf => return it hash
        if (subtree_leaves_len == 1)
        {
            hasher->cpu_hash_one(curr_leaves_buf_ptr, leaf_size, curr_digests_buf_ptr);
            memcpy(curr_cap_buf_ptr, curr_digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            hasher->cpu_hash_one(curr_leaves_buf_ptr, leaf_size, curr_digests_buf_ptr);
            hasher->cpu_hash_one(curr_leaves_buf_ptr + leaf_size, leaf_size, curr_digests_buf_ptr + HASH_SIZE_U64);
            hasher->cpu_hash_two(curr_digests_buf_ptr, curr_digests_buf_ptr + HASH_SIZE_U64, curr_cap_buf_ptr);
            continue;
        }

        // 2. compute leaf hashes
        u64 *digests_curr_ptr = curr_digests_buf_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;
#pragma omp parallel for
        for (u32 i = 0; i < subtree_leaves_len; i++)
        {
#ifdef RUST_POSEIDON
            ext_poseidon_hash_or_noop(digests_curr_ptr + (i * HASH_SIZE_U64), curr_leaves_buf_ptr + (i * leaf_size), leaf_size);
#else
            hasher->cpu_hash_one(curr_leaves_buf_ptr + (i * leaf_size), leaf_size, digests_curr_ptr + (i * HASH_SIZE_U64));
#endif
        }

        // 3. compute internal hashes
        u32 r = (u32)log2(subtree_leaves_len) - 1;
        u32 last_index = subtree_digests_len - subtree_leaves_len;

        for (; r > 0; r--)
        {
            last_index -= (1 << r);
            // printf("CPU round %d Last idx %d\n", r, last_index);
            u64 *curr_digests_buf_ptr2 = curr_digests_buf_ptr + last_index * HASH_SIZE_U64;

#pragma omp parallel for
            for (int idx = 0; idx < (1 << r); idx++)
            {
                u32 left_idx = 2 * (idx + 1) + last_index;
                u32 right_idx = left_idx + 1;
                u64 *left_ptr = curr_digests_buf_ptr2 + (left_idx * HASH_SIZE_U64);
                u64 *right_ptr = curr_digests_buf_ptr2 + (right_idx * HASH_SIZE_U64);
                // printf("%lu %lu\n", *left_ptr, *right_ptr);
#ifdef RUST_POSEIDON
                ext_poseidon_hash_of_two(curr_digests_buf_ptr2 + (idx * HASH_SIZE_U64), left_ptr, right_ptr);
#else
                hasher->cpu_hash_two(left_ptr, right_ptr, curr_digests_buf_ptr2 + (idx * HASH_SIZE_U64));
#endif
            }
        }

        // 4. compute cap hashes
#ifdef RUST_POSEIDON
        ext_poseidon_hash_of_two(curr_cap_buf_ptr, curr_digests_buf_ptr, curr_digests_buf_ptr + HASH_SIZE_U64);
#else
        hasher->cpu_hash_two(curr_digests_buf_ptr, curr_digests_buf_ptr + HASH_SIZE_U64, curr_cap_buf_ptr);
#endif

    } // end for k

    delete hasher;
}
