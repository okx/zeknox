// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "poseidon/poseidon.hpp"
#include "poseidon/poseidon_permutation.hpp"
#include "ff/goldilocks.hpp"
#include <cstring>

const static i64 MDS_FREQ_BLOCK_ONE[3] = {16, 32, 16};
const static i64 MDS_FREQ_BLOCK_TWO[6] = {2, -1, -4, 1, 16, 1};
const static i64 MDS_FREQ_BLOCK_THREE[3] = {-1, -8, 2};

inline u128 PoseidonPermutation::mds_row_shf(u64 r, u64 *v)
{
    assert(r < SPONGE_WIDTH);
    // The values of `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are
    // known to be small, so we can accumulate all the products for
    // each row and reduce just once at the end (done by the
    // caller).

    u128 res = 0;
    for (int i = 0; i < 12; i++)
    {
        res += (u128)(v[(i + r) % SPONGE_WIDTH]) * (u128)MDS_MATRIX_CIRC[i];
    }
    res += (u128)v[r] * (u128)MDS_MATRIX_DIAG[r];

    return res;
}

inline GoldilocksField PoseidonPermutation::sbox_monomial(const GoldilocksField &x)
{
    GoldilocksField x2 = x * x;
    GoldilocksField x4 = x2 * x2;
    GoldilocksField x3 = x * x2;
    return x3 * x4;
}

inline void PoseidonPermutation::sbox_layer(GoldilocksField *inout)
{
    for (int i = 0; i < 12; i++)
    {
        inout[i] = sbox_monomial(inout[i]);
    }
}

inline void PoseidonPermutation::mds_layer(GoldilocksField *inout)
{
    u64 s[SPONGE_WIDTH] = {0};

    for (u32 r = 0; r < SPONGE_WIDTH; r++)
    {
        s[r] = inout[r].to_noncanonical_u64();
    }

    for (u32 r = 0; r < SPONGE_WIDTH; r++)
    {
        u128 sum = mds_row_shf(r, s);
        u64 sum_lo = (u64)sum;
        u32 sum_hi = (u32)(sum >> 64);
        inout[r] = GoldilocksField::from_noncanonical_u96(sum_lo, sum_hi);
    }
}

inline void PoseidonPermutation::constant_layer(GoldilocksField *inout, u32 *round_ctr)
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        u64 round_constant = ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * (*round_ctr)];
        inout[i] = inout[i].add_canonical_u64(round_constant);
    }
}

// Arrays of size SPIONGE_WIDTH
inline void PoseidonPermutation::full_rounds(GoldilocksField *inout, u32 *round_ctr)
{
    for (u32 k = 0; k < HALF_N_FULL_ROUNDS; k++)
    {
        constant_layer(inout, round_ctr);
        sbox_layer(inout);
        mds_layer_fast(inout);
        // mds_layer(inout);
        *round_ctr += 1;
    }
}

inline void PoseidonPermutation::partial_rounds_naive(GoldilocksField *inout, u32 *round_ctr)
{
    for (u64 k = 0; k < N_PARTIAL_ROUNDS; k++)
    {
        constant_layer(inout, round_ctr);
        inout[0] = sbox_monomial(inout[0]);
        mds_layer(inout);
        *round_ctr += 1;
    }
}

// NOTE: inout should point to an array of size SPIONGE_WIDTH
inline void PoseidonPermutation::poseidon_naive(GoldilocksField *inout)
{
    u32 round_ctr = 0;

    full_rounds(inout, &round_ctr);
    // print_perm(inout, SPONGE_WIDTH);
    partial_rounds_naive(inout, &round_ctr);
    // print_perm(inout, SPONGE_WIDTH);
    full_rounds(inout, &round_ctr);
    // print_perm(inout, SPONGE_WIDTH);
    assert(round_ctr == N_ROUNDS);
}

inline void PoseidonPermutation::partial_first_constant_layer(GoldilocksField *inout)
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        inout[i] = inout[i] + inout[i].from_canonical_u64(FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
    }
}

inline void PoseidonPermutation::mds_partial_layer_init(GoldilocksField *inout)
{
    GoldilocksField result[SPONGE_WIDTH] = {GoldilocksField::zero()};

    for (u32 r = 1; r < SPONGE_WIDTH; r++)
    {
        for (u32 c = 1; c < SPONGE_WIDTH; c++)
        {
            GoldilocksField t = GoldilocksField::from_canonical_u64(
                FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1]);
            result[c] = result[c] + inout[r] * t;
        }
    }
    for (u32 i = 1; i < SPONGE_WIDTH; i++)
    {
        inout[i] = result[i];
    }
}

inline void PoseidonPermutation::add_u160_u128(u128 *x_lo, u32 *x_hi, u128 y)
{
    u128 M = (u128)-1;
    if (*x_lo > M - y)
    {
        *x_lo = *x_lo + y;
        *x_hi = *x_hi + 1;
    }
    else
    {
        *x_lo = *x_lo + y;
    }
}

inline GoldilocksField PoseidonPermutation::reduce_u160(u128 n_lo, u32 n_hi)
{
    u64 n_lo_hi = (u64)(n_lo >> 64);
    u64 n_lo_lo = (u64)n_lo;
    u64 reduced_hi = GoldilocksField::from_noncanonical_u96(n_lo_hi, n_hi).to_noncanonical_u64();
    u128 reduced128 = (((u128)reduced_hi) << 64) + (u128)n_lo_lo;
    return GoldilocksField::from_noncanonical_u128(reduced128);
}

inline void PoseidonPermutation::mds_partial_layer_fast(GoldilocksField *inout, u32 r)
{
    // u169 accumulator
    u128 d_sum_lo = 0;
    u32 d_sum_hi = 0;
    for (u32 i = 1; i < 12; i++)
    {
        u128 t = (u128)FAST_PARTIAL_ROUND_W_HATS[r][i - 1];
        u128 si = (u128)inout[i].to_noncanonical_u64();
        add_u160_u128(&d_sum_lo, &d_sum_hi, si * t);
    }
    u128 s0 = (u128)inout[0].to_noncanonical_u64();
    u128 mds0to0 = (u128)(MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0]);
    add_u160_u128(&d_sum_lo, &d_sum_hi, s0 * mds0to0);
    GoldilocksField d = reduce_u160(d_sum_lo, d_sum_hi);

    GoldilocksField result[SPONGE_WIDTH] = {GoldilocksField::zero()};
    result[0] = d;
    for (u32 i = 1; i < SPONGE_WIDTH; i++)
    {
        GoldilocksField t = GoldilocksField::from_canonical_u64(FAST_PARTIAL_ROUND_VS[r][i - 1]);
        result[i] = inout[i].multiply_accumulate(inout[0], t);
    }
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        inout[i] = result[i];
    }
}

inline void PoseidonPermutation::partial_rounds(GoldilocksField *inout, u32 *round_ctr)
{
    partial_first_constant_layer(inout);
    mds_partial_layer_init(inout);

    for (u32 i = 0; i < N_PARTIAL_ROUNDS; i++)
    {
        inout[0] = sbox_monomial(inout[0]);
        inout[0] = inout[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
        mds_partial_layer_fast(inout, i);
    }
    *round_ctr += N_PARTIAL_ROUNDS;
}

inline void PoseidonPermutation::block1(const i64 *x, const i64 *y, i64 *z)
{
    z[0] = x[0] * y[0] + x[1] * y[2] + x[2] * y[1];
    z[1] = x[0] * y[1] + x[1] * y[0] + x[2] * y[2];
    z[2] = x[0] * y[2] + x[1] * y[1] + x[2] * y[0];
}

inline void PoseidonPermutation::block2(const i64 *x, const i64 *y, i64 *z)
{
    i64 x0r = x[0];
    i64 x0i = x[1];
    i64 x1r = x[2];
    i64 x1i = x[3];
    i64 x2r = x[4];
    i64 x2i = x[5];

    i64 y0r = y[0];
    i64 y0i = y[1];
    i64 y1r = y[2];
    i64 y1i = y[3];
    i64 y2r = y[4];
    i64 y2i = y[5];

    i64 x0s = x0r + x0i;
    i64 x1s = x1r + x1i;
    i64 x2s = x2r + x2i;
    i64 y0s = y0r + y0i;
    i64 y1s = y1r + y1i;
    i64 y2s = y2r + y2i;

    i64 m00 = x0r * y0r;
    i64 m01 = x0i * y0i;
    i64 m10 = x1r * y2r;
    i64 m11 = x1i * y2i;
    i64 m20 = x2r * y1r;
    i64 m21 = x2i * y1i;
    i64 z0r = (m00 - m01) + (x1s * y2s - m10 - m11) + (x2s * y1s - m20 - m21);
    i64 z0i = (x0s * y0s - m00 - m01) + (-m10 + m11) + (-m20 + m21);
    z[0] = z0r;
    z[1] = z0i;

    m00 = x0r * y1r;
    m01 = x0i * y1i;
    m10 = x1r * y0r;
    m11 = x1i * y0i;
    m20 = x2r * y2r;
    m21 = x2i * y2i;
    i64 z1r = (m00 - m01) + (m10 - m11) + (x2s * y2s - m20 - m21);
    i64 z1i = (x0s * y1s - m00 - m01) + (x1s * y0s - m10 - m11) + (-m20 + m21);
    z[2] = z1r;
    z[3] = z1i;

    m00 = x0r * y2r;
    m01 = x0i * y2i;
    m10 = x1r * y1r;
    m11 = x1i * y1i;
    m20 = x2r * y0r;
    m21 = x2i * y0i;
    i64 z2r = (m00 - m01) + (m10 - m11) + (m20 - m21);
    i64 z2i = (x0s * y2s - m00 - m01) + (x1s * y1s - m10 - m11) + (x2s * y0s - m20 - m21);
    z[4] = z2r;
    z[5] = z2i;
}

inline void PoseidonPermutation::block3(const i64 *x, const i64 *y, i64 *z)
{
    z[0] = x[0] * y[0] - x[1] * y[2] - x[2] * y[1];
    z[1] = x[0] * y[1] + x[1] * y[0] - x[2] * y[2];
    z[2] = x[0] * y[2] + x[1] * y[1] + x[2] * y[0];
}

inline void PoseidonPermutation::fft2_real(const u64 *x, i64 *z)
{
    z[0] = (i64)x[0] + (i64)x[1];
    z[1] = (i64)x[0] - (i64)x[1];
}

inline void PoseidonPermutation::ifft2_real_unreduced(const i64 *y, u64 *z)
{
    z[0] = (u64)(y[0] + y[1]);
    z[1] = (u64)(y[0] - y[1]);
}

inline void PoseidonPermutation::fft4_real(const u64 *x, i64 *y)
{
    u64 xa[2] = {x[0], x[2]};
    u64 xb[2] = {x[1], x[3]};
    i64 z[4];

    fft2_real(xa, &z[0]);
    fft2_real(xb, &z[2]);

    y[0] = z[0] + z[2];
    y[1] = z[1];
    y[2] = -z[3];
    y[3] = z[0] - z[2];
}

inline void PoseidonPermutation::ifft4_real_unreduced(const i64 *y, u64 *x)
{
    i64 za[2] = {y[0] + y[3], y[1]};
    i64 zb[2] = {y[0] - y[3], -y[2]};
    u64 xa[2], xb[2];

    ifft2_real_unreduced(za, xa);
    ifft2_real_unreduced(zb, xb);

    x[0] = xa[0];
    x[1] = xb[0];
    x[2] = xa[1];
    x[3] = xb[1];
}

inline u64 PoseidonPermutation::reduce96(u128 val)
{
    gl64_t g = gl64_t::from_noncanonical_u96(val);
    return g.get_val();
}

inline u64 PoseidonPermutation::reduce128(u128 val)
{
    gl64_t g = gl64_t::from_noncanonical_u128(val);
    return g.get_val();
}

inline void PoseidonPermutation::mds_multiply_freq(u64 *inout)
{
    u64 sa[4] = {inout[0], inout[3], inout[6], inout[9]};
    u64 sb[4] = {inout[1], inout[4], inout[7], inout[10]};
    u64 sc[4] = {inout[2], inout[5], inout[8], inout[11]};

    i64 u[12];
    fft4_real(sa, &u[0]);
    fft4_real(sb, &u[4]);
    fft4_real(sc, &u[8]);

    // printf("&&& %lX %lX %lX %lX\n", u[0], u[1], u[2], u[3]);

    i64 ua[3] = {u[0], u[4], u[8]};
    i64 ub[6] = {u[1], u[2], u[5], u[6], u[9], u[10]};
    i64 uc[4] = {u[3], u[7], u[11]};
    i64 v[12];

    block1(ua, MDS_FREQ_BLOCK_ONE, v);
    block2(ub, MDS_FREQ_BLOCK_TWO, &v[3]);
    block3(uc, MDS_FREQ_BLOCK_THREE, &v[9]);

    i64 va[4] = {v[0], v[3], v[4], v[9]};
    i64 vb[4] = {v[1], v[5], v[6], v[10]};
    i64 vc[4] = {v[2], v[7], v[8], v[11]};

    ifft4_real_unreduced(va, sa);
    ifft4_real_unreduced(vb, sb);
    ifft4_real_unreduced(vc, sc);

    inout[0] = sa[0];
    inout[1] = sb[0];
    inout[2] = sc[0];
    inout[3] = sa[1];
    inout[4] = sb[1];
    inout[5] = sc[1];
    inout[6] = sa[2];
    inout[7] = sb[2];
    inout[8] = sc[2];
    inout[9] = sa[3];
    inout[10] = sb[3];
    inout[11] = sc[3];
}

inline void PoseidonPermutation::mds_layer_fast(GoldilocksField *inout)
{
    u64 state_l[SPONGE_WIDTH], state_h[SPONGE_WIDTH];

    for (uint32_t r = 0; r < SPONGE_WIDTH; r++)
    {
        state_h[r] = inout[r].get_val() >> 32;
        state_l[r] = inout[r].get_val() & 0xFFFFFFFF;
    }

    mds_multiply_freq(state_h);
    mds_multiply_freq(state_l);

    // MDS_MATRIX_DIAG[0] is 8, so we shift inout[0] by 3
    u128 s0 = ((u128)inout[0].get_val()) << 3;

    for (uint32_t r = 0; r < SPONGE_WIDTH; r++)
    {
        u128 s = (u128)state_l[r] + (((u128)state_h[r]) << 32);
        inout[r] = reduce96(s);
    }

    GoldilocksField s0u64 = reduce96(s0);
    inout[0] = inout[0] + s0u64;
}

// Arrys of size SPIONGE_WIDTH
void PoseidonPermutation::poseidon(GoldilocksField *inout)
{
    u32 round_ctr = 0;

    full_rounds(inout, &round_ctr);
    partial_rounds(inout, &round_ctr);
    full_rounds(inout, &round_ctr);
}

PoseidonPermutation::PoseidonPermutation()
{
    for (u64 i = 0; i < WIDTH; i++)
    {
        this->state[i] = GoldilocksField::zero();
    }
}

void PoseidonPermutation::set_from_slice(GoldilocksField *elts, u64 len, u64 start_idx)
{
    // assert(start_idx + len <= WIDTH);
    for (u64 i = 0; i < len; i++)
    {
        this->state[start_idx + i] = elts[i];
    }
}

void PoseidonPermutation::get_state_as_canonical_u64(u64 *out)
{
    assert(out != 0);
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        out[i] = this->state[i].to_noncanonical_u64();
    }
}

GoldilocksField *PoseidonPermutation::get_state()
{
    return this->state;
}

void PoseidonPermutation::set_state(u32 idx, GoldilocksField val)
{
    assert(idx < SPONGE_WIDTH);
    this->state[idx] = val;
}

void PoseidonPermutation::permute()
{
    // poseidon_naive(this->state);
    poseidon(this->state);
}

void PoseidonHasher::cpu_hash_one(u64 *input, u64 input_count, u64 *digest)
{
    PoseidonPermutation::cpu_hash_one_with_permutation_template<PoseidonPermutation>(input, input_count, digest);
}

void PoseidonHasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
    PoseidonPermutation::cpu_hash_two_with_permutation_template<PoseidonPermutation>(digest_left, digest_right, digest);
}
