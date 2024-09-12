#ifndef __POSEIDON2_HPP__
#define __POSEIDON2_HPP__

#include "types/int_types.h"
#include "utils/cuda_utils.cuh"
#include "types/monty31.hpp"
#include <random>

#ifdef USE_CUDA
#include "types/gl64_t.cuh"
#include "types/bb31_t.cuh"
#include "poseidon/poseidon_permutation.cuh"
#else
#include "types/goldilocks.hpp"
#include "types/babybear.hpp"
#include "poseidon/poseidon.hpp"
#include "poseidon/poseidon_permutation.hpp"
#include <cstring>
#endif

#include "merkle/hasher.hpp"

class Poseidon2Hasher : public Hasher
{

public:
#ifdef USE_CUDA
    __host__ static void cpu_hash_one(u64 *input, u64 size, u64 *output);
    __host__ static void cpu_hash_two(u64 *input1, u64 *input2, u64 *output);
    __device__ static void gpu_hash_one(gl64_t *input, u32 size, gl64_t *output);
    __device__ static void gpu_hash_two(gl64_t *input1, gl64_t *input2, gl64_t *output);
#else
    static void cpu_hash_one(u64 *input, u64 size, u64 *output);
    static void cpu_hash_two(u64 *input1, u64 *input2, u64 *output);
#endif
};

#ifdef USE_CUDA
class Poseidon2PermutationGPU : public PoseidonPermutationGPU
{
public:
    DEVICE void permute();
};
#else  // USE_CUDA
class Poseidon2Permutation : public PoseidonPermutation
{
public:
    void permute();
};
#endif // USE_CUDA

#include <stdio.h>

DEVICE INLINE u64 gcd(u64 a, u64 b)
{
    while (b != 0)
    {
        u64 t = b;
        b = a % b;
        a = t;
    }
    return a;
}

DEVICE INLINE u64 ilog2(u64 x)
{
    u64 result = 0;
    while (x >>= 1)
    {
        result++;
    }
    return result;
}

template <typename F>
inline void poseidon2_round_numbers_128(u64 width, u64 d, u64 *round_f, u64 *round_p)
{
    // Start by checking that d is a valid permutation.
    assert(gcd(d, F::ORDER_U64 - 1) == 1);

    // Next compute the number of bits in p.
    u64 prime_bit_number = ilog2(F::ORDER_U64) + 1;

    if (prime_bit_number == 31)
    {
        *round_f = 8;
        if (width == 16)
        {
            switch (d)
            {
            case 3:
                *round_p = 20;
                break;
            case 5:
                *round_p = 14;
                break;
            case 7:
            case 9:
            case 11:
                *round_p = 13;
                break;
            default:
                assert(0);
            }
        }
        else if (width == 24)
        {
            switch (d)
            {
            case 3:
                *round_p = 23;
                break;
            case 5:
                *round_p = 22;
                break;
            case 7:
            case 9:
            case 11:
                *round_p = 21;
                break;
            default:
                assert(0);
            }
        }
        else
        {
            assert(0);
        }
    }
    else if (prime_bit_number == 64)
    {
        *round_f = 8;
        if (width == 8)
        {
            switch (d)
            {
            case 3:
                *round_p = 41;
                break;
            case 5:
                *round_p = 27;
                break;
            case 7:
                *round_p = 22;
                break;
            case 9:
                *round_p = 19;
                break;
            case 11:
                *round_p = 17;
                break;
            default:
                assert(0);
            }
        }
        else if (width == 12)
        {
            switch (d)
            {
            case 3:
                *round_p = 42;
                break;
            case 5:
                *round_p = 27;
                break;
            case 7:
                *round_p = 22;
                break;
            case 9:
                *round_p = 20;
                break;
            case 11:
                *round_p = 18;
                break;
            default:
                assert(0);
            }
        }
        else if (width == 16)
        {
            switch (d)
            {
            case 3:
                *round_p = 42;
                break;
            case 5:
                *round_p = 27;
                break;
            case 7:
                *round_p = 22;
                break;
            case 9:
                *round_p = 20;
                break;
            case 11:
                *round_p = 18;
                break;
            default:
                assert(0);
            }
        }
        else
        {
            assert(0);
        }
    }
    else
    {
        assert(0);
    }
}

/*
 * Poseidon2 implementation as in Plonky3.
 */
template <typename F, typename MdsLight, typename Diffusion, const u64 WIDTH, const u64 D>
class Poseidon2
{
private:
    /// The number of external rounds.
    u64 rounds_f;

    /// The external round constants.
    F *external_constants;

    /// The linear layer used in External Rounds. Should be either MDS or a
    /// circulant matrix based off an MDS matrix of size 4.
    MdsLight external_linear_layer;

    /// The number of internal rounds.
    u64 rounds_p;

    /// The internal round constants.
    F *internal_constants;

    /// The linear layer used in internal rounds (only needs diffusion property, not MDS).
    Diffusion internal_linear_layer;

    DEVICE INLINE void exp_const_u64(F *x)
    {
        switch (D)
        {
        case 1:
            break;
        case 2:
            *x = (*x) * (*x);
            break;
        case 3:
            *x = (*x) * (*x) * (*x);
            break;
        case 4:
        {
            F x2 = (*x) * (*x);
            *x = x2 * x2;
        }
        break;
        case 5:
        {
            F x2 = (*x) * (*x);
            F x4 = x2 * x2;
            *x = (*x) * x4;
        }
        break;
        case 6:
        {
            F x2 = (*x) * (*x);
            F x4 = x2 * x2;
            *x = x2 * x4;
        }
        break;
        case 7:
        {
            F x2 = (*x) * (*x);
            F x4 = x2 * x2;
            F x3 = (*x) * x2;
            *x = x3 * x4;
        }
        break;
        default:
            printf("Invalid power value!\n");
            assert(0);
        }
    }

    DEVICE INLINE void add_rc(F *state, F *rc)
    {
        for (u64 i = 0; i < WIDTH; i++)
        {
            state[i] += rc[i];
        }
    }

    DEVICE INLINE void sbox_p(F *state)
    {
        exp_const_u64(state);
    }

    DEVICE INLINE void sbox(F *state)
    {
        for (u64 i = 0; i < WIDTH; i++)
        {
            exp_const_u64(&state[i]);
        }
    }

public:
    DEVICE Poseidon2(u64 rounds_f, u64 rounds_p, F *external_constants, F *internal_constants, MdsLight external_linear_layer, Diffusion internal_linear_layer)
    {
        this->rounds_f = rounds_f;
        this->rounds_p = rounds_p;
        this->external_linear_layer = external_linear_layer;
        this->internal_linear_layer = internal_linear_layer;
        this->external_constants = external_constants;
        this->internal_constants = internal_constants;
    }

    /*
    // TODO - revise
    static DEVICE Poseidon2 new_from_rng_128(
        MdsLight external_linear_layer,
        Diffusion internal_linear_layer)
    {
        u64 rounds_f, rounds_p;
        poseidon2_round_numbers_128<F>(WIDTH, D, &rounds_f, &rounds_p);

        xoroshiro128plus_engine eng;

        std::random_device dev{};
        eng.seed([&dev]()
                 { return dev(); });
        std::uniform_int_distribution<u64> dist(0, F::ORDER_U64 - 1);
        F external_constants[rounds_f * WIDTH];
        for (u64 i = 0; i < rounds_f * WIDTH; i++)
        {
            external_constants[i] = F(dist(eng));
        }
        F internal_constants[rounds_p];
        for (u64 i = 0; i < rounds_p; i++)
        {
            internal_constants[i] = F(dist(eng));
        }

        return Poseidon2(
            rounds_f,
            external_constants,
            external_linear_layer,
            rounds_p,
            internal_constants,
            internal_linear_layer);
    }
    */

    DEVICE void
    permute_mut(F *state)
    {
        // The initial linear layer.
        this->external_linear_layer.permute_mut(state);

        // The first half of the external rounds.
        u64 rounds_f_half = rounds_f / 2;
        for (u64 r = 0; r < rounds_f_half; r++)
        {
            add_rc(state, &(this->external_constants[r * WIDTH]));
            sbox(state);
            this->external_linear_layer.permute_mut(state);
        }

        // The internal rounds.
        for (u64 r = 0; r < rounds_p; r++)
        {
            state[0] += this->internal_constants[r];
            sbox_p(&state[0]);
            this->internal_linear_layer.permute_mut(state);
        }

        // The second half of the external rounds.
        for (u64 r = rounds_f_half; r < rounds_f; r++)
        {
            add_rc(state, &(this->external_constants[r * WIDTH]));
            sbox(state);
            this->external_linear_layer.permute_mut(state);
        }
    }
};

template <typename F>
DEVICE INLINE void apply_mat4(F *x)
{
    F t01 = x[0] + x[1];
    F t23 = x[2] + x[3];
    F t0123 = t01 + t23;
    F t01123 = t0123 + x[1];
    F t01233 = t0123 + x[3];
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233 + x[0] + x[0]; // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123 + x[2] + x[2]; // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01;         // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23;         // x[0] + x[1] + 2*x[2] + 3*x[3]
}

template <typename F>
DEVICE INLINE void apply_hl_mat4(F *x)
{
    F t0 = x[0] + x[1];
    F t1 = x[2] + x[3];
    F t2 = x[1] + x[1] + t1;
    F t3 = x[3] + x[3] + t0;
    F t4 = t1 + t1 + t1 + t1 + t3;
    F t5 = t0 + t0 + t0 + t0 + t2;
    F t6 = t3 + t5;
    F t7 = t2 + t4;
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

template <typename F>
class MdsPerm4
{
public:
    static DEVICE INLINE void permute_mut(F *state) {};
};

template <typename F>
class HLMDSMat4 : public MdsPerm4<F>
{
public:
    static DEVICE INLINE void permute_mut(F *state)
    {
        apply_hl_mat4<F>(state);
    }
};

template <typename F>
class MDSMat4 : public MdsPerm4<F>
{
public:
    static DEVICE INLINE void permute_mut(F *state)
    {
        apply_mat4<F>(state);
    }
};

template <typename F, typename Mat4, const u64 WIDTH>
DEVICE INLINE void mds_light_permutation(F *state)
{
    // Mat4 mdsmat;
    switch (WIDTH)
    {
    case 2:
    {
        F sum2 = state[0] + state[1];
        state[0] += sum2;
        state[1] += sum2;
    }
    break;

    case 3:
    {
        F sum3 = state[0] + state[1] + state[2];
        state[0] += sum3;
        state[1] += sum3;
        state[2] += sum3;
    }
    break;

    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
    {
        // First, we apply M_4 to each consecutive four elements of the state.
        // In Appendix B's terminology, this replaces each x_i with x_i'.
        for (u64 i = 0; i < WIDTH; i += 4)
        {
            // Would be nice to find a better way to do this.
            // F state_4;
            // std::memcpy(state_4, &state[i], 4 * sizeof(F));
            // mdsmat.permute_mut(&state[i]);
            Mat4::permute_mut(&state[i]);
            // std::memcpy(&state[i], state_4, 4 * sizeof(F));
        }
        // Now, we apply the outer circulant matrix (to compute the y_i values).

        // We first precompute the four sums of every four elements.
        F sums[4] = {F::Zero()};
        for (u64 k = 0; k < 4; k++)
        {
            for (u64 i = 0; i < WIDTH; i += 4)
            {
                sums[k] += state[i + k];
            }
        }

        // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
        // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
        for (u64 i = 0; i < WIDTH; i++)
        {
            state[i] += sums[i % 4];
        }
    }
    break;

    default:
        printf("Unsupported width %lu\n", WIDTH);
        assert(0);
    }
}

template <typename F, const u64 WIDTH>
class Poseidon2ExternalMatrixHL
{
public:
    static DEVICE INLINE void permute_mut(F *state)
    {
        mds_light_permutation<F, HLMDSMat4<F>, WIDTH>(state);
    }
};

template <typename F, const u64 WIDTH>
class Poseidon2ExternalMatrixGeneral
{
public:
    static DEVICE INLINE void permute_mut(F *state)
    {
        mds_light_permutation<F, MDSMat4<F>, WIDTH>(state);
    }
};

template <typename F, const u64 WIDTH>
DEVICE INLINE void matmul_internal_(F *state, F *mat_internal_diag_m_1)
{
    F sum = state[0];
    for (u64 i = 1; i < WIDTH; i++)
    {
        sum += state[i];
    }

    for (u64 i = 0; i < WIDTH; i++)
    {
        state[i] *= mat_internal_diag_m_1[i];
        state[i] += sum;
    }
}

template <const u64 WIDTH>
DEVICE void permute_state(BabyBearField *state, const u32 *internal_diag_shifts)
{
    u64 part_sum = state[1].value();
    for (u64 i = 2; i < WIDTH; i++)
    {
        part_sum += state[i].value();
    }
    u64 full_sum = part_sum + state[0].value();
    u64 s0 = part_sum + (u64)(BabyBearField::Zero() - state[0]).value();

    state[0] = BabyBearField(monty_reduce<BabyBearParameters>(s0), true);

    for (u64 i = 0; i < WIDTH - 1; i++)
    {
        u64 si = full_sum + ((u64)state[i + 1].value() << internal_diag_shifts[i]);
        state[i + 1] = BabyBearField(monty_reduce<BabyBearParameters>(si), true);
    }
}

template <typename F, const u64 WIDTH>
class DiffusionMatrixGoldilocks
{
public:
    static DEVICE INLINE void permute_mut(F *state);
};

template <const u64 WIDTH>
class BabyBearDiffusionMatrix
{
private:
    u32 INTERNAL_DIAG_SHIFTS_16[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
    u32 INTERNAL_DIAG_SHIFTS_24[23] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23};

public:
    DEVICE INLINE void permute_mut(BabyBearField *state)
    {
        switch (WIDTH)
        {
        case 16:
            return permute_state<16>(state, INTERNAL_DIAG_SHIFTS_16);
        case 24:
            return permute_state<24>(state, INTERNAL_DIAG_SHIFTS_24);
        default:
            return;
        }
    }
};

/*
class BabyBearDiffusionMatrix16 : public BabyBearDiffusionMatrix<16>
{
private:
    u32 INTERNAL_DIAG_SHIFTS[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};

    BabyBearField INTERNAL_DIAG_MONTY[16] = {
        // BabyBearField::ORDER_U32 - 2,
        2013265919,
        1,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
        1 << 8,
        1 << 9,
        1 << 10,
        1 << 11,
        1 << 12,
        1 << 13,
        1 << 15,
    };

public:
    DEVICE INLINE void permute_mut(BabyBearField *state)
    {
        permute_state<16>(state, INTERNAL_DIAG_SHIFTS);
    }
};

class BabyBearDiffusionMatrix24 : public BabyBearDiffusionMatrix<24>
{
private:
    u32 INTERNAL_DIAG_SHIFTS[23] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23};

    BabyBearField INTERNAL_DIAG_MONTY[24] = {
        // BabyBearField::ORDER_U32 - 2,
        2013265919,
        1,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
        1 << 8,
        1 << 9,
        1 << 10,
        1 << 11,
        1 << 12,
        1 << 13,
        1 << 14,
        1 << 15,
        1 << 16,
        1 << 18,
        1 << 19,
        1 << 20,
        1 << 21,
        1 << 22,
        1 << 23,
    };

public:
    DEVICE INLINE void permute_mut(BabyBearField *state)
    {
        permute_state<24>(state, INTERNAL_DIAG_SHIFTS);
    }
};
*/

#ifdef USE_CUDA
__device__ void hl_poseidon2_goldilocks_width_8(gl64_t *input);
#else
void hl_poseidon2_goldilocks_width_8(GoldilocksField *input);
#endif

#endif // __POSEIDON2_HPP__