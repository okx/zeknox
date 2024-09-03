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

    GoldilocksField* get_state();

    void permute();
};

#endif // __POSEIDON_PERMUTATION_HPP__