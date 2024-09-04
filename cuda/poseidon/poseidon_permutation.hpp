#ifndef __POSEIDON_PERMUTATION_HPP__
#define __POSEIDON_PERMUTATION_HPP__

#include "poseidon/poseidon.hpp"
#include "poseidon/goldilocks.hpp"

class PoseidonPermutation : public PoseidonPermutationVirtual
{
private:
    GoldilocksField state[WIDTH];

    static inline u128 mds_row_shf(u64 r, u64 *v);

    static inline GoldilocksField sbox_monomial(const GoldilocksField &x);

    inline void sbox_layer(GoldilocksField *inout);

    void mds_layer(GoldilocksField *inout);

    inline void constant_layer(GoldilocksField *inout, u32 *round_ctr);

    inline void full_rounds(GoldilocksField *inout, u32 *round_ctr);

    void partial_rounds_naive(GoldilocksField *inout, u32 *round_ctr);

    void poseidon_naive(GoldilocksField *inout);

    void partial_first_constant_layer(GoldilocksField *state);

    void mds_partial_layer_init(GoldilocksField *state);

    static inline void add_u160_u128(u128 *x_lo, u32 *x_hi, u128 y);

    static inline GoldilocksField reduce_u160(u128 n_lo, u32 n_hi);

    void mds_partial_layer_fast(GoldilocksField *state, u32 r);

    void partial_rounds(GoldilocksField *state, u32 *round_ctr);

    void poseidon(GoldilocksField *inout);

public:
    PoseidonPermutation();

    void set_from_slice(GoldilocksField *elts, u64 len, u64 start_idx);

    void get_state_as_canonical_u64(u64 *out);

    void set_state(u32 idx, GoldilocksField val);

    GoldilocksField *get_state();

    void permute();

    template<class P>
    static void cpu_hash_one_with_permutation_template(u64 *input, u64 input_count, u64 *output)
    {
        // special cases
        if (input_count < NUM_HASH_OUT_ELTS)
        {
            std::memcpy(output, input, input_count * sizeof(u64));
            std::memset(output + input_count, 0, (NUM_HASH_OUT_ELTS - input_count) * sizeof(u64));
            return;
        }
        if (input_count == NUM_HASH_OUT_ELTS)
        {
            std::memcpy(output, input, input_count * sizeof(u64));
            return;
        }

        // prepare input
        GoldilocksField *in = new GoldilocksField[input_count];
        for (u32 i = 0; i < input_count; i++)
        {
            in[i] = GoldilocksField(input[i]);
        }

        // permutation
        P perm = P();
        // absorb all input chunks
        u64 idx = 0;
        while (idx < input_count)
        {
            perm.set_from_slice(in + idx, MIN(PoseidonPermutation::RATE, (input_count - idx)), 0);
            perm.permute();
            idx += PoseidonPermutation::RATE;
        }

        // set output
        u64 out[12];
        perm.get_state_as_canonical_u64(out);
        std::memcpy(output, out, NUM_HASH_OUT_ELTS * sizeof(u64));
        delete[] in;
    }

    template<class P>
    static void cpu_hash_two_with_permutation_template(u64 *digest_left, u64 *digest_right, u64 *digest)
    {
        GoldilocksField in1[4] = {digest_left[0], digest_left[1], digest_left[2], digest_left[3]};
        GoldilocksField in2[4] = {digest_right[0], digest_right[1], digest_right[2], digest_right[3]};

        P perm = P();
        perm.set_from_slice(in1, NUM_HASH_OUT_ELTS, 0);
        perm.set_from_slice(in2, NUM_HASH_OUT_ELTS, NUM_HASH_OUT_ELTS);

        perm.permute();

        u64 out[12];
        perm.get_state_as_canonical_u64(out);
        std::memcpy(digest, out, NUM_HASH_OUT_ELTS * sizeof(u64));
    }
};

#endif // __POSEIDON_PERMUTATION_HPP__