#include "poseidon.hpp"
#include "poseidon.h"
#include <stdlib.h>

/// A one-way compression function which takes two ~256 bit inputs and returns a ~256 bit output.
HashOut compress(HashOut x, HashOut y)
{
    // TODO: With some refactoring, this function could be implemented as
    // hash_n_to_m_no_pad(chain(x.elements, y.elements), NUM_HASH_OUT_ELTS).

    assert(x.n_elements == NUM_HASH_OUT_ELTS);
    assert(y.n_elements == NUM_HASH_OUT_ELTS);
    assert(PoseidonPermutation::RATE >= NUM_HASH_OUT_ELTS);

    PoseidonPermutation perm = PoseidonPermutation();
    perm.set_from_slice(x.elements, x.n_elements, 0);
    perm.set_from_slice(y.elements, y.n_elements, NUM_HASH_OUT_ELTS);

    perm.permute();

    return perm.squeeze(NUM_HASH_OUT_ELTS);
}

/// Hash a message without any padding step. Note that this can enable length-extension attacks.
/// However, it is still collision-resistant in cases where the input has a fixed length.
HashOut hash_n_to_m_no_pad(GoldilocksField *inputs, u64 num_inputs, u64 num_outputs)
{
    PoseidonPermutation perm = PoseidonPermutation();

    // Absorb all input chunks.
    u64 idx = 0;
    while (idx < num_inputs)
    {
        perm.set_from_slice(inputs + idx, MIN(PoseidonPermutation::RATE, (num_inputs - idx)), 0);
        perm.permute();
        idx += PoseidonPermutation::RATE;
    }

    // Squeeze until we have the desired number of outputs.
    assert(num_outputs == NUM_HASH_OUT_ELTS);
    return perm.squeeze(NUM_HASH_OUT_ELTS);
}

HashOut hash_no_pad(GoldilocksField *inputs, u64 n_inputs)
{
    if (n_inputs <= NUM_HASH_OUT_ELTS) {
        return HashOut(inputs, n_inputs);
    }

    return hash_n_to_m_no_pad(inputs, n_inputs, NUM_HASH_OUT_ELTS);
}

HashOut two_to_one(HashOut left, HashOut right)
{
    return compress(left, right);
}

void test_MDS(GoldilocksField *inputs, u64 num_inputs)
{
    PoseidonPermutation perm = PoseidonPermutation();
    perm.set_from_slice(inputs, num_inputs, 0);
    // perm.mds_layer(perm.state, inputs);    
}

void compute_hash_leaf(u64 *digest, u64 *data, u32 data_size)
{
    GoldilocksField *in = (GoldilocksField *)malloc(data_size * sizeof(GoldilocksField));
    for (u32 i = 0; i < data_size; i++)
    {
        in[i] = GoldilocksField(data[i]);
    }
    HashOut out = hash_no_pad(in, data_size);
    for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        digest[i] = out.elements[i].get_val();
    }
    free(in);
}

void compute_hash_of_two(u64 *digest, u64 *digest_left, u64 *digest_right)
{
    HashOut in_l = HashOut(digest_left, NUM_HASH_OUT_ELTS);
    HashOut in_r = HashOut(digest_right, NUM_HASH_OUT_ELTS);
    HashOut out = two_to_one(in_l, in_r);
    for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        digest[i] = out.elements[i].get_val();
    }
}
