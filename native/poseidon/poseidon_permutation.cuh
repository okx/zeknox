// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __POSEIDON_V2_CUH__
#define __POSEIDON_V2_CUH__

#include "types/int_types.h"
#include "ff/gl64_t.cuh"
#include "utils/cuda_utils.cuh"
#include "merkle/hasher.hpp"
#include "poseidon/poseidon.hpp"

#define MIN(x, y) (x < y) ? x : y

#define SPONGE_RATE 8
#define SPONGE_CAPACITY 4
#define SPONGE_WIDTH 12

#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL 8
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS 30
#define MAX_WIDTH 12

#ifdef USE_CUDA

class PoseidonPermutationGPU : public PoseidonPermutationGPUVirtual
{
private:
    DEVICE INLINE static gl64_t reduce128(u128 x);

    DEVICE INLINE static gl64_t reduce_u160(u128 n_lo, u32 n_hi);

    DEVICE INLINE static void add_u160_u128(u128 *x_lo, u32 *x_hi, u128 y);

    DEVICE INLINE static gl64_t from_noncanonical_u96(gl64_t n_lo, gl64_t n_hi);

    DEVICE INLINE static gl64_t from_noncanonical_u128(u128 n);

    DEVICE INLINE void mds_partial_layer_fast(gl64_t *state, u32 r);

    DEVICE INLINE gl64_t mds_row_shf(u32 r, gl64_t *v);

    DEVICE INLINE void mds_layer(gl64_t *state, gl64_t *result);

    DEVICE INLINE void constant_layer(gl64_t *state, u32 *round_ctr);

    DEVICE INLINE static gl64_t sbox_monomial(const gl64_t &x);

    DEVICE INLINE void sbox_layer(gl64_t *state);

    DEVICE INLINE void full_rounds(gl64_t *state, u32 *round_ctr);

    DEVICE INLINE void partial_rounds_naive(gl64_t *state, u32 *round_ctr);

    DEVICE INLINE void partial_rounds(gl64_t *state, u32 *round_ctr);

    DEVICE INLINE gl64_t *poseidon_naive(gl64_t *input);

    DEVICE INLINE gl64_t *poseidon(gl64_t *input);

protected:
    gl64_t state[SPONGE_WIDTH];

public:
    DEVICE PoseidonPermutationGPU();

    DEVICE void set_from_slice(gl64_t *elts, u32 len, u32 start_idx);

    DEVICE void set_from_slice_stride(gl64_t *elts, u32 len, u32 start_idx, u32 stride);

    DEVICE void get_state_as_canonical_u64(u64 *out);

    DEVICE void set_state(u32 idx, gl64_t val);

    DEVICE void permute();

    DEVICE gl64_t* get_state() { return state; };

    DEVICE gl64_t *squeeze(u32 size);

    template<class P>
    DEVICE INLINE static void gpu_hash_one_with_permutation_template(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
    {
        /*
         * NOTE: to avoid a branch, we assume the input size is > NUM_HASH_OUT_ELTS. For inputs with size < NUM_HASH_OUT_ELTS,
         * this function produces incorrect output. This case is filered out by an assert in Merkle Tree building functions.
         */
#if 1
        if (num_inputs <= NUM_HASH_OUT_ELTS)
        {
            u32 i = 0;
            for (; i < num_inputs; i++)
            {
                hash[i] = inputs[i];
            }
            for (; i < NUM_HASH_OUT_ELTS; i++)
            {
                hash[i].make_zero();
            }
        }
        else
        {
#endif
            P perm = P();
            for (u32 idx = 0; idx < num_inputs; idx += SPONGE_RATE)
            {
                perm.set_from_slice(inputs + idx, MIN(SPONGE_RATE, num_inputs - idx), 0);
                perm.permute();
            }
            gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
            for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
            {
                hash[i] = ret[i];
            }
#if 1
        }
#endif
    };

    template<class P>
    DEVICE INLINE static void gpu_hash_two_with_permutation_template(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
    {
        P perm = P();
        perm.set_from_slice(hash1, NUM_HASH_OUT_ELTS, 0);
        perm.set_from_slice(hash2, NUM_HASH_OUT_ELTS, NUM_HASH_OUT_ELTS);
        perm.permute();
        gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
        for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
        {
            hash[i] = ret[i];
        }
    }
};

#ifdef DEBUG
DEVICE void print_perm(gl64_t *data, int cnt);
#endif

#endif // USE_CUDA

#endif // __POSEIDON_V2_CUH__
