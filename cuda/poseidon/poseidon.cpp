#include "poseidon/poseidon.hpp"
#include "poseidon/poseidon_permutation.hpp"
#include "poseidon/goldilocks.hpp"
#include <cstring>

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

void PoseidonPermutation::mds_layer(GoldilocksField *inout)
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
        mds_layer(inout);
        *round_ctr += 1;
    }
}

void PoseidonPermutation::partial_rounds_naive(GoldilocksField *inout, u32 *round_ctr)
{
    for (u64 k = 0; k < N_PARTIAL_ROUNDS; k++)
    {
        constant_layer(inout, round_ctr);
        inout[0] = sbox_monomial(inout[0]);
        mds_layer(inout);
        *round_ctr += 1;
    }
}

// Arrys of size SPIONGE_WIDTH
void PoseidonPermutation::poseidon_naive(GoldilocksField *inout)
{
    u32 round_ctr = 0;

    full_rounds(inout, &round_ctr);
    // print_perm(state, SPONGE_WIDTH);
    partial_rounds_naive(inout, &round_ctr);
    // print_perm(state, SPONGE_WIDTH);
    full_rounds(inout, &round_ctr);
    // print_perm(state, SPONGE_WIDTH);
    assert(round_ctr == N_ROUNDS);
}

void PoseidonPermutation::partial_first_constant_layer(GoldilocksField *state)
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + state[i].from_canonical_u64(FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
    }
}

void PoseidonPermutation::mds_partial_layer_init(GoldilocksField *state)
{
    GoldilocksField result[SPONGE_WIDTH] = {GoldilocksField::Zero()};

    for (u32 r = 1; r < SPONGE_WIDTH; r++)
    {
        for (u32 c = 1; c < SPONGE_WIDTH; c++)
        {
            GoldilocksField t = GoldilocksField::from_canonical_u64(
                FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1]);
            result[c] = result[c] + state[r] * t;
        }
    }
    for (u32 i = 1; i < SPONGE_WIDTH; i++)
    {
        state[i] = result[i];
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

void PoseidonPermutation::mds_partial_layer_fast(GoldilocksField *state, u32 r)
{
    // u169 accumulator
    u128 d_sum_lo = 0;
    u32 d_sum_hi = 0;
    for (u32 i = 1; i < 12; i++)
    {
        u128 t = (u128)FAST_PARTIAL_ROUND_W_HATS[r][i - 1];
        u128 si = (u128)state[i].to_noncanonical_u64();
        add_u160_u128(&d_sum_lo, &d_sum_hi, si * t);
    }
    u128 s0 = (u128)state[0].to_noncanonical_u64();
    u128 mds0to0 = (u128)(MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0]);
    add_u160_u128(&d_sum_lo, &d_sum_hi, s0 * mds0to0);
    GoldilocksField d = reduce_u160(d_sum_lo, d_sum_hi);

    GoldilocksField result[SPONGE_WIDTH] = {GoldilocksField::Zero()};
    result[0] = d;
    for (u32 i = 1; i < SPONGE_WIDTH; i++)
    {
        GoldilocksField t = GoldilocksField::from_canonical_u64(FAST_PARTIAL_ROUND_VS[r][i - 1]);
        result[i] = state[i].multiply_accumulate(state[0], t);
    }
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = result[i];
    }
}

void PoseidonPermutation::partial_rounds(GoldilocksField *state, u32 *round_ctr)
{
    partial_first_constant_layer(state);
    mds_partial_layer_init(state);

    for (u32 i = 0; i < N_PARTIAL_ROUNDS; i++)
    {
        state[0] = sbox_monomial(state[0]);
        state[0] = state[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
        mds_partial_layer_fast(state, i);
    }
    *round_ctr += N_PARTIAL_ROUNDS;
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
        this->state[i] = GoldilocksField::Zero();
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
        out[i] = state[i].to_noncanonical_u64();
    }
}

GoldilocksField* PoseidonPermutation::get_state()
{
    return state;
}

void PoseidonPermutation::set_state(u32 idx, GoldilocksField val)
{
    assert(idx < SPONGE_WIDTH);
    state[idx] = val;
}

void PoseidonPermutation::permute()
{
    poseidon_naive(this->state);
    // poseidon(this->state);
}

void PoseidonHasher::cpu_hash_one(u64 *input, u64 input_count, u64 *digest)
{
    PoseidonPermutation::cpu_hash_one_with_permutation_template<PoseidonPermutation>(input, input_count, digest);
}

void PoseidonHasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
    PoseidonPermutation::cpu_hash_two_with_permutation_template<PoseidonPermutation>(digest_left, digest_right, digest);
}
