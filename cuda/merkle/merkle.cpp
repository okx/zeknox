#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include <cstring>

#include "int_types.h"
#include "merkle.h"
#include "merkle_private.h"
#include "merkle_c.h"
#include "poseidon.h"
#include "poseidon2.h"
#include "poseidon_bn128.h"
#include "keccak.h"
#include "monolith.h"

#ifndef __AVX512__
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

/*
 * Selectors of CPU hash functions.
 */
inline void cpu_hash_one_ptr(u64 *input, u32 size, u64 *data, uint64_t hash_type)
{
    switch (hash_type)
    {
    case HashPoseidon:
        return cpu_poseidon_hash_one(input, size, data);
    case HashKeccak:
        return cpu_keccak_hash_one(input, size, data);
    case HashPoseidonBN128:
        return cpu_poseidon_bn128_hash_one(input, size, data);
    case HashPoseidon2:
        return cpu_poseidon2_hash_one(input, size, data);
    case HashMonolith:
        return cpu_monolith_hash_one(input, size, data);
    default:
        return;
    }
}

inline void cpu_hash_two_ptr(u64 *hash1, u64 *hash2, u64 *hash, uint64_t hash_type)
{
    switch (hash_type)
    {
    case HashPoseidon:
        return cpu_poseidon_hash_two(hash1, hash2, hash);
    case HashKeccak:
        return cpu_keccak_hash_two(hash1, hash2, hash);
    case HashPoseidonBN128:
        return cpu_poseidon_bn128_hash_two(hash1, hash2, hash);
    case HashPoseidon2:
        return cpu_poseidon2_hash_two(hash1, hash2, hash);
    case HashMonolith:
        return cpu_monolith_hash_two(hash1, hash2, hash);
    default:
        return;
    }
}

#else // #ifndef __AVX512__

#include "poseidon.h"

inline void cpu_hash_one_ptr(u64 *input, u32 size, u64 *data, uint64_t hash_type)
{
    cpu_poseidon_hash_one(input, size, data);
}
inline void cpu_hash_two_ptr(u64 *hash1, u64 *hash2, u64 *hash, uint64_t hash_type)
{
    cpu_poseidon_hash_two(hash1, hash2, hash);
}

#endif // #ifndef __AVX512__

void fill_digests_buf_linear_cpu(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    void *leaves_buf_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type)
{
    if (cap_buf_size == leaves_buf_size)
    {
#pragma omp parallel for
        for (u64 i = 0; i < leaves_buf_size; i++)
        {
            cpu_hash_one_ptr((u64 *)leaves_buf_ptr + (i * leaf_size), leaf_size, (u64 *)digests_buf_ptr + i * HASH_SIZE_U64, hash_type);
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
        u64 *it_leaves_buf_ptr = (u64 *)leaves_buf_ptr + k * subtree_leaves_len * leaf_size;
        u64 *it_digests_buf_ptr = (u64 *)digests_buf_ptr + k * subtree_digests_len * HASH_SIZE_U64;
        u64 *it_cap_buf_ptr = (u64 *)cap_buf_ptr + k * HASH_SIZE_U64;

        // if one leaf => return it hash
        if (subtree_leaves_len == 1)
        {
            cpu_hash_one_ptr(it_leaves_buf_ptr, leaf_size, it_digests_buf_ptr, hash_type);
            memcpy(it_cap_buf_ptr, it_digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            cpu_hash_one_ptr(it_leaves_buf_ptr, leaf_size, it_digests_buf_ptr, hash_type);
            cpu_hash_one_ptr(it_leaves_buf_ptr + leaf_size, leaf_size, it_digests_buf_ptr + HASH_SIZE_U64, hash_type);
            cpu_hash_two_ptr(it_digests_buf_ptr, it_digests_buf_ptr + HASH_SIZE_U64, it_cap_buf_ptr, hash_type);
            continue;
        }

        // 2. compute leaf hashes
        u64 *digests_curr_ptr = it_digests_buf_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;
#pragma omp parallel for
        for (u32 i = 0; i < subtree_leaves_len; i++)
        {
            cpu_hash_one_ptr(it_leaves_buf_ptr + (i * leaf_size), leaf_size, digests_curr_ptr + (i * HASH_SIZE_U64), hash_type);
        }

        // 3. compute internal hashes
        u32 r = (u32)log2(subtree_leaves_len) - 1;
        u32 last_index = subtree_digests_len - subtree_leaves_len;

        for (; r > 0; r--)
        {
            last_index -= (1 << r);
            // printf("CPU round %d Last idx %d\n", r, last_index);
            u64 *digests_buf_ptr2 = it_digests_buf_ptr + last_index * HASH_SIZE_U64;

#pragma omp parallel for
            for (int idx = 0; idx < (1 << r); idx++)
            {
                u32 left_idx = 2 * (idx + 1) + last_index;
                u32 right_idx = left_idx + 1;
                u64 *left_ptr = digests_buf_ptr2 + (left_idx * HASH_SIZE_U64);
                u64 *right_ptr = digests_buf_ptr2 + (right_idx * HASH_SIZE_U64);
                cpu_hash_two_ptr(left_ptr, right_ptr, digests_buf_ptr2 + (idx * HASH_SIZE_U64), hash_type);
            }
        }

        // 4. compute cap hashes
        cpu_hash_two_ptr(it_digests_buf_ptr, it_digests_buf_ptr + HASH_SIZE_U64, it_cap_buf_ptr, hash_type);

    } // end for k
}

#ifdef __AVX512__

#define RATE 8
#define CAPACITY 4
#define SPONGE_WIDTH (RATE + CAPACITY)
#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL (2 * HALF_N_FULL_ROUNDS)
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS (N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS)

#include "goldilocks_base_field_avx.hpp"
#include "goldilocks_base_field_avx512.hpp"
#include "poseidon_goldilocks_constants.hpp"
#include <immintrin.h>

const Goldilocks::Element Goldilocks::CQ = {(uint64_t)0x00000000FFFFFFFFLL};
const Goldilocks::Element Goldilocks::ZR = {(uint64_t)0x0000000000000000LL};
const Goldilocks::Element Goldilocks::TWO32 = {0x0000000100000000LL};

inline void pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
}

inline void pow7_avx(__m256i &st0, __m256i &st1, __m256i &st2)
{
    __m256i pw2_0, pw2_1, pw2_2;
    Goldilocks::square_avx(pw2_0, st0);
    Goldilocks::square_avx(pw2_1, st1);
    Goldilocks::square_avx(pw2_2, st2);
    __m256i pw4_0, pw4_1, pw4_2;
    Goldilocks::square_avx(pw4_0, pw2_0);
    Goldilocks::square_avx(pw4_1, pw2_1);
    Goldilocks::square_avx(pw4_2, pw2_2);
    __m256i pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_avx(pw3_0, pw2_0, st0);
    Goldilocks::mult_avx(pw3_1, pw2_1, st1);
    Goldilocks::mult_avx(pw3_2, pw2_2, st2);

    Goldilocks::mult_avx(st0, pw3_0, pw4_0);
    Goldilocks::mult_avx(st1, pw3_1, pw4_1);
    Goldilocks::mult_avx(st2, pw3_2, pw4_2);
}

inline void add_avx(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    __m256i c0, c1, c2;
    Goldilocks::load_avx(c0, &(C_[0]));
    Goldilocks::load_avx(c1, &(C_[4]));
    Goldilocks::load_avx(c2, &(C_[8]));
    Goldilocks::add_avx(st0, st0, c0);
    Goldilocks::add_avx(st1, st1, c1);
    Goldilocks::add_avx(st2, st2, c2);
}

inline void add_avx_small(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    __m256i c0, c1, c2;
    Goldilocks::load_avx(c0, &(C_small[0]));
    Goldilocks::load_avx(c1, &(C_small[4]));
    Goldilocks::load_avx(c2, &(C_small[8]));

    Goldilocks::add_avx_b_small(st0, st0, c0);
    Goldilocks::add_avx_b_small(st1, st1, c1);
    Goldilocks::add_avx_b_small(st2, st2, c2);
}

inline void pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
{
    __m512i pw2_0, pw2_1, pw2_2;
    Goldilocks::square_avx512(pw2_0, st0);
    Goldilocks::square_avx512(pw2_1, st1);
    Goldilocks::square_avx512(pw2_2, st2);
    __m512i pw4_0, pw4_1, pw4_2;
    Goldilocks::square_avx512(pw4_0, pw2_0);
    Goldilocks::square_avx512(pw4_1, pw2_1);
    Goldilocks::square_avx512(pw4_2, pw2_2);
    __m512i pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_avx512(pw3_0, pw2_0, st0);
    Goldilocks::mult_avx512(pw3_1, pw2_1, st1);
    Goldilocks::mult_avx512(pw3_2, pw2_2, st2);

    Goldilocks::mult_avx512(st0, pw3_0, pw4_0);
    Goldilocks::mult_avx512(st1, pw3_1, pw4_1);
    Goldilocks::mult_avx512(st2, pw3_2, pw4_2);
}

inline void add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    __m512i c0 = _mm512_set4_epi64(C_[3].fe, C_[2].fe, C_[1].fe, C_[0].fe);
    __m512i c1 = _mm512_set4_epi64(C_[7].fe, C_[6].fe, C_[5].fe, C_[4].fe);
    __m512i c2 = _mm512_set4_epi64(C_[11].fe, C_[10].fe, C_[9].fe, C_[8].fe);
    Goldilocks::add_avx512(st0, st0, c0);
    Goldilocks::add_avx512(st1, st1, c1);
    Goldilocks::add_avx512(st2, st2, c2);
}

inline void add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    __m512i c0 = _mm512_set4_epi64(C_small[3].fe, C_small[2].fe, C_small[1].fe, C_small[0].fe);
    __m512i c1 = _mm512_set4_epi64(C_small[7].fe, C_small[6].fe, C_small[5].fe, C_small[4].fe);
    __m512i c2 = _mm512_set4_epi64(C_small[11].fe, C_small[10].fe, C_small[9].fe, C_small[8].fe);

    Goldilocks::add_avx512_b_c(st0, st0, c0);
    Goldilocks::add_avx512_b_c(st1, st1, c1);
    Goldilocks::add_avx512_b_c(st2, st2, c2);
}

void hash_full_result_avx(Goldilocks::Element *state, const Goldilocks::Element *input)
{
    // const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    // std::memcpy(state, input, length);
    __m256i st0, st1, st2;
    Goldilocks::load_avx(st0, &(state[0]));
    Goldilocks::load_avx(st1, &(state[4]));
    Goldilocks::load_avx(st2, &(state[8]));
    add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx(st0, st1, st2);
        add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH]));
        Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_avx(st0, st1, st2);
    add_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    Goldilocks::mmult_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::P_[0]));

    Goldilocks::store_avx(&(state[0]), st0);
    Goldilocks::Element state0_ = state[0];
    Goldilocks::Element state0;

    __m256i mask = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0);
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state0 = state0_;
        pow7(state0);
        state0 = state0 + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        state0_ = state0 * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
        st0 = _mm256_and_si256(st0, mask);
        state0_ = state0_ + Goldilocks::dot_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
        __m256i scalar1 = _mm256_set1_epi64x(state0.fe);
        __m256i w0, w1, w2, s0, s1, s2;
        Goldilocks::load_avx(s0, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        Goldilocks::load_avx(s1, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 4]));
        Goldilocks::load_avx(s2, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 8]));
        Goldilocks::mult_avx(w0, scalar1, s0);
        Goldilocks::mult_avx(w1, scalar1, s1);
        Goldilocks::mult_avx(w2, scalar1, s2);
        Goldilocks::add_avx(st0, st0, w0);
        Goldilocks::add_avx(st1, st1, w1);
        Goldilocks::add_avx(st2, st2, w2);
        state0 = state0 + PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
    }
    Goldilocks::store_avx(&(state[0]), st0);
    state[0] = state0_;
    Goldilocks::load_avx(st0, &(state[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx(st0, st1, st2);
        add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_avx(st0, st1, st2);
    Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));

    Goldilocks::store_avx(&(state[0]), st0);
    Goldilocks::store_avx(&(state[4]), st1);
    Goldilocks::store_avx(&(state[8]), st2);
}

void linear_hash_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(&output[size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        /*
        else
        {
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
        }
        */

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        // memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        hash_full_result_avx(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}

void hash_full_result_avx512(Goldilocks::Element *state, const Goldilocks::Element *input)
{
    // const int length = 2 * SPONGE_WIDTH * sizeof(Goldilocks::Element);
    // std::memcpy(state, input, length);
    __m512i st0, st1, st2;
    Goldilocks::load_avx512(st0, &(state[0]));
    Goldilocks::load_avx512(st1, &(state[8]));
    Goldilocks::load_avx512(st2, &(state[16]));
    add_avx512_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx512(st0, st1, st2);
        add_avx512_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH])); // rick
        Goldilocks::mmult_avx512_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_avx512(st0, st1, st2);
    add_avx512(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    Goldilocks::mmult_avx512(st0, st1, st2, &(PoseidonGoldilocksConstants::P_[0]));

    Goldilocks::store_avx512(&(state[0]), st0);
    Goldilocks::Element s04_[2] = {state[0], state[4]};
    Goldilocks::Element s04[2];

    __m512i mask = _mm512_set_epi64(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0); // rick, not better to define where u use it?
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        s04[0] = s04_[0];
        s04[1] = s04_[1];
        pow7(s04[0]);
        pow7(s04[1]);
        s04[0] = s04[0] + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        s04[1] = s04[1] + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        s04_[0] = s04[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
        s04_[1] = s04[1] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
        st0 = _mm512_and_si512(st0, mask); // rick, do we need a new one?
        Goldilocks::Element aux[2];
        Goldilocks::dot_avx512(aux, st0, st1, st2, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
        s04_[0] = s04_[0] + aux[0];
        s04_[1] = s04_[1] + aux[1];
        __m512i scalar1 = _mm512_set_epi64(s04[1].fe, s04[1].fe, s04[1].fe, s04[1].fe, s04[0].fe, s04[0].fe, s04[0].fe, s04[0].fe);
        __m512i w0, w1, w2;

        const Goldilocks::Element *auxS = &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]);
        __m512i s0 = _mm512_set4_epi64(auxS[3].fe, auxS[2].fe, auxS[1].fe, auxS[0].fe);
        __m512i s1 = _mm512_set4_epi64(auxS[7].fe, auxS[6].fe, auxS[5].fe, auxS[4].fe);
        __m512i s2 = _mm512_set4_epi64(auxS[11].fe, auxS[10].fe, auxS[9].fe, auxS[8].fe);

        Goldilocks::mult_avx512(w0, scalar1, s0);
        Goldilocks::mult_avx512(w1, scalar1, s1);
        Goldilocks::mult_avx512(w2, scalar1, s2);
        Goldilocks::add_avx512(st0, st0, w0);
        Goldilocks::add_avx512(st1, st1, w1);
        Goldilocks::add_avx512(st2, st2, w2);
        s04[0] = s04[0] + PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
        s04[1] = s04[1] + PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
    }

    Goldilocks::store_avx512(&(state[0]), st0);
    state[0] = s04_[0];
    state[4] = s04_[1];
    Goldilocks::load_avx512(st0, &(state[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx512(st0, st1, st2);
        add_avx512_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        Goldilocks::mmult_avx512_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_avx512(st0, st1, st2);
    Goldilocks::mmult_avx512_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));

    Goldilocks::store_avx512(&(state[0]), st0);
    Goldilocks::store_avx512(&(state[8]), st1);
    Goldilocks::store_avx512(&(state[16]), st2);
}

void linear_hash_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[2 * SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(output + size, 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        std::memcpy(output + CAPACITY, input + size, size * sizeof(Goldilocks::Element));
        std::memset(output + CAPACITY + size, 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + 2 * RATE, 0, 2 * CAPACITY * sizeof(Goldilocks::Element));
        }
        /*
        else
        {
            std::memcpy(state + 2 * RATE, state, 2 * CAPACITY * sizeof(Goldilocks::Element));
        }
        */

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        // memset(state, 0, 2 * RATE * sizeof(Goldilocks::Element));

        if (n <= 4)
        {
            std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
            std::memcpy(state + 4, input + size + (size - remaining), n * sizeof(Goldilocks::Element));
        }
        else
        {
            std::memcpy(state, input + (size - remaining), 4 * sizeof(Goldilocks::Element));
            std::memcpy(state + 4, input + size + (size - remaining), 4 * sizeof(Goldilocks::Element));
            std::memcpy(state + 8, input + (size - remaining) + 4, (n - 4) * sizeof(Goldilocks::Element));
            std::memcpy(state + 12, input + size + (size - remaining) + 4, (n - 4) * sizeof(Goldilocks::Element));
        }

        hash_full_result_avx512(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, 2 * CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, 2 * CAPACITY * sizeof(Goldilocks::Element));
    }
}

void fill_digests_buf_linear_cpu_avx(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    void *leaves_buf_ptr,
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type)
{
    if (hash_type != HashPoseidon)
    {
        assert(!"Only Poseidon is available under __AVX512__!");
    }

    if (cap_buf_size == leaves_buf_size)
    {
#pragma omp parallel for
        for (u64 i = 0; i < leaves_buf_size; i++)
        {
            linear_hash_avx((Goldilocks::Element *)digests_buf_ptr + i * HASH_SIZE_U64, (Goldilocks::Element *)leaves_buf_ptr + (i * leaf_size), leaf_size);
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
        Goldilocks::Element *it_leaves_buf_ptr = (Goldilocks::Element *)leaves_buf_ptr + k * subtree_leaves_len * leaf_size;
        Goldilocks::Element *it_digests_buf_ptr = (Goldilocks::Element *)digests_buf_ptr + k * subtree_digests_len * HASH_SIZE_U64;
        Goldilocks::Element *it_cap_buf_ptr = (Goldilocks::Element *)cap_buf_ptr + k * HASH_SIZE_U64;

        // if one leaf => return it hash
        if (subtree_leaves_len == 1)
        {
            linear_hash_avx(it_digests_buf_ptr, it_leaves_buf_ptr, leaf_size);
            std::memcpy(it_cap_buf_ptr, it_digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            linear_hash_avx(it_digests_buf_ptr, it_leaves_buf_ptr, leaf_size);
            linear_hash_avx(it_digests_buf_ptr + HASH_SIZE_U64, it_leaves_buf_ptr + leaf_size, leaf_size);
            linear_hash_avx(it_cap_buf_ptr, it_digests_buf_ptr, 2 * HASH_SIZE_U64);
            continue;
        }

        // 2. compute leaf hashes
        Goldilocks::Element *digests_curr_ptr = it_digests_buf_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;
#pragma omp parallel for
        for (u32 i = 0; i < subtree_leaves_len; i++)
        {
            linear_hash_avx(digests_curr_ptr + (i * HASH_SIZE_U64), it_leaves_buf_ptr + (i * leaf_size), leaf_size);
        }

        // 3. compute internal hashes
        u32 r = (u32)log2(subtree_leaves_len) - 1;
        u32 last_index = subtree_digests_len - subtree_leaves_len;

        for (; r > 0; r--)
        {
            last_index -= (1 << r);
            Goldilocks::Element *digests_buf_ptr2 = it_digests_buf_ptr + last_index * HASH_SIZE_U64;

#pragma omp parallel for
            for (int idx = 0; idx < (1 << r); idx++)
            {
                u32 left_idx = 2 * (idx + 1) + last_index;
                Goldilocks::Element *left_ptr = digests_buf_ptr2 + (left_idx * HASH_SIZE_U64);
                linear_hash_avx(digests_buf_ptr2 + (idx * HASH_SIZE_U64), left_ptr, 2 * HASH_SIZE_U64);
            }
        }

        // 4. compute cap hashes
        linear_hash_avx(it_cap_buf_ptr, it_digests_buf_ptr, 2 * HASH_SIZE_U64);

    } // end for k
}

void fill_digests_buf_linear_cpu_avx512(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    void *leaves_buf_ptr,
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type)
{
    if (hash_type != HashPoseidon)
    {
        assert(!"Only Poseidon is available under __AVX512__!");
    }

    if (cap_buf_size == leaves_buf_size)
    {
#pragma omp parallel for
        for (u64 i = 0; i < leaves_buf_size; i += 2)
        {
            linear_hash_avx512((Goldilocks::Element *)digests_buf_ptr + i * HASH_SIZE_U64, (Goldilocks::Element *)leaves_buf_ptr + (i * leaf_size), leaf_size);
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
        Goldilocks::Element *it_leaves_buf_ptr = (Goldilocks::Element *)leaves_buf_ptr + k * subtree_leaves_len * leaf_size;
        Goldilocks::Element *it_digests_buf_ptr = (Goldilocks::Element *)digests_buf_ptr + k * subtree_digests_len * HASH_SIZE_U64;
        Goldilocks::Element *it_cap_buf_ptr = (Goldilocks::Element *)cap_buf_ptr + k * HASH_SIZE_U64;

        // if one leaf => return it hash
        if (subtree_leaves_len == 1)
        {
            linear_hash_avx(it_digests_buf_ptr, it_leaves_buf_ptr, leaf_size);
            std::memcpy(it_cap_buf_ptr, it_digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            linear_hash_avx512(it_digests_buf_ptr, it_leaves_buf_ptr, leaf_size);
            linear_hash_avx(it_cap_buf_ptr, it_digests_buf_ptr, 2 * HASH_SIZE_U64);
            continue;
        }

        // 2. compute leaf hashes
        Goldilocks::Element *digests_curr_ptr = it_digests_buf_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;
#pragma omp parallel for
        for (u32 i = 0; i < subtree_leaves_len; i += 2)
        {
            linear_hash_avx512(digests_curr_ptr + (i * HASH_SIZE_U64), it_leaves_buf_ptr + (i * leaf_size), leaf_size);
        }

        // 3. compute internal hashes
        u32 r = (u32)log2(subtree_leaves_len) - 1;
        u32 last_index = subtree_digests_len - subtree_leaves_len;

        for (; r > 0; r--)
        {
            last_index -= (1 << r);
            // printf("CPU round %d Last idx %d\n", r, last_index);
            Goldilocks::Element *digests_buf_ptr2 = it_digests_buf_ptr + last_index * HASH_SIZE_U64;

#pragma omp parallel for
            for (int idx = 0; idx < (1 << r); idx += 2)
            {
                u32 left_idx = 2 * (idx + 1) + last_index;
                Goldilocks::Element *left_ptr = digests_buf_ptr2 + (left_idx * HASH_SIZE_U64);
                linear_hash_avx512(digests_buf_ptr2 + (idx * HASH_SIZE_U64), left_ptr, 2 * HASH_SIZE_U64);
            }
        }

        // 4. compute cap hashes
        linear_hash_avx(it_cap_buf_ptr, it_digests_buf_ptr, 2 * HASH_SIZE_U64);

    } // end for k
}

#else // __AVX512__

void fill_digests_buf_linear_cpu_avx512(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    void *leaves_buf_ptr,
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type)
{
    assert(!"Function fill_digests_buf_linear_cpu_avx512() is available only under __AVX512__!");
}

#endif // __AVX512__

/**
 * End of library code.
 */

// #define TESTING
#ifdef TESTING

#define LEAF_SIZE_U64 84

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

void generate_random_leaves(u64 *leaves, u32 n_leaves, u32 leaf_size)
{
    srand(time(NULL));

#pragma omp parallel for
    for (u32 i = 0; i < n_leaves * leaf_size; i++)
    {
        u32 r = rand();
        leaves[i] = ((u64)r << 32) + r * 88958514;
    }
}

void generate_const_leaves(u64 *leaves, u32 n_leaves, u32 leaf_size)
{
#pragma omp parallel for
    for (u32 i = 0; i < n_leaves * leaf_size; i++)
    {
        leaves[i] = (u64)i * 2147483647;
    }
}

int main()
{
    u64 cap_h = 0;
    u64 n_caps = (1 << cap_h);
    u64 n_leaves = (1 << 10);
    u64 n_digests = 2 * (n_leaves - n_caps);

    u64 *global_leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));
    u64 *global_digests_buf = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *global_cap_buf = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *global_digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *global_cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));

    generate_const_leaves(global_leaves_buf, n_leaves, LEAF_SIZE_U64);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    fill_digests_buf_linear_cpu(
        global_digests_buf,
        global_cap_buf,
        global_leaves_buf,
        n_digests,
        n_caps,
        n_leaves,
        LEAF_SIZE_U64,
        cap_h,
        HashPoseidon);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Merkle Tree building time (no AVX): %lu ms\n", t / 1000);

    printf("Root is: %lx %lx %lx %lx\n", global_cap_buf[0], global_cap_buf[1], global_cap_buf[2], global_cap_buf[3]);

    /*
    for (u64 i = 0; i < n_digests * HASH_SIZE_U64; i++)
    {
        printf("%lx ", global_digests_buf[i]);
        if (i % 4 == 3) {
            printf("\n");
        }
    }
    */

    gettimeofday(&start, NULL);
    fill_digests_buf_linear_cpu_avx512(
        global_digests_buf2,
        global_cap_buf2,
        global_leaves_buf,
        n_digests,
        n_caps,
        n_leaves,
        LEAF_SIZE_U64,
        cap_h,
        HashPoseidon);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Merkle Tree building time (AVX512): %lu ms\n", t / 1000);
    // printf("Merkle Tree building time (AVX): %lu ms\n", t / 1000);

    printf("Root is: %lx %lx %lx %lx\n", global_cap_buf2[0], global_cap_buf2[1], global_cap_buf2[2], global_cap_buf2[3]);

    /*
    for (u64 i = 0; i < n_digests * HASH_SIZE_U64; i++)
    {
        printf("%lx ", global_digests_buf2[i]);
        if (i % 4 == 3) {
            printf("\n");
        }
    }
    */

    // compare
    bool same = true;
    for (u64 i = 0; i < n_digests * HASH_SIZE_U64; i++)
    {
        if (global_digests_buf[i] != global_digests_buf2[i])
        {
            same = false;
            break;
        }
    }
    for (u64 i = 0; same && i < n_caps * HASH_SIZE_U64; i++)
    {
        if (global_cap_buf[i] != global_cap_buf2[i])
        {
            same = false;
            break;
        }
    }
    if (same)
    {
        printf("Ok. The MTs are the same.\n");
    }
    else
    {
        printf("FAILURE! The MTs are different.\n");
    }

    free(global_leaves_buf);
    free(global_digests_buf);
    free(global_cap_buf);
    free(global_digests_buf2);
    free(global_cap_buf2);

    return 0;
}
#endif // TESTING
