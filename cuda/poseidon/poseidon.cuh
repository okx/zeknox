#ifndef __POSEIDON_V2_CUH__
#define __POSEIDON_V2_CUH__

#include "int_types.h"
#include "gl64_t.cuh"
#include "cuda_utils.cuh"

#define MIN(x, y) (x < y) ? x : y

#define NUM_HASH_OUT_ELTS 4

// From poseidon.rs
#define SPONGE_RATE 8
#define SPONGE_CAPACITY 4
// #define SPONGE_WIDTH SPONGE_RATE + SPONGE_CAPACITY
#define SPONGE_WIDTH 12

// The number of full rounds and partial rounds is given by the
// calc_round_numbers.py script. They happen to be the same for both
// width 8 and width 12 with s-box x^7.
//
// NB: Changing any of these values will require regenerating all of
// the precomputed constant arrays in this file.
#define HALF_N_FULL_ROUNDS 4
// #define N_FULL_ROUNDS_TOTAL 2 * HALF_N_FULL_ROUNDS
#define N_FULL_ROUNDS_TOTAL 8
#define N_PARTIAL_ROUNDS 22
// #define N_ROUNDS N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS
#define N_ROUNDS 30
#define MAX_WIDTH 12
// we only have width 8 and 12, and 12 is bigger. :)

extern __device__ u32 GPU_MDS_MATRIX_CIRC[12];
extern __device__ u32 GPU_MDS_MATRIX_DIAG[12];
extern __device__ u64 GPU_ALL_ROUND_CONSTANTS[MAX_WIDTH * N_ROUNDS];

class PoseidonPermutationGPU
{
public:
    __device__ inline static gl64_t reduce128(uint128_t x);

    __device__ inline static gl64_t from_noncanonical_u96(gl64_t n_lo, gl64_t n_hi);
    
    __device__ gl64_t mds_row_shf(u32 r, gl64_t *v);

    __device__ void mds_layer(gl64_t *state, gl64_t *result);

    __device__ inline void constant_layer(gl64_t *state, u32 *round_ctr);

    __device__ static inline gl64_t sbox_monomial(const gl64_t &x);

    __device__ inline void sbox_layer(gl64_t *state);

    __device__ inline void full_rounds(gl64_t *state, u32 *round_ctr);

    __device__ void partial_rounds_naive(gl64_t *state, u32 *round_ctr);

    // Arrys of size SPIONGE_WIDTH
    __device__ gl64_t *poseidon_naive(gl64_t *input);

    gl64_t state[SPONGE_WIDTH];

public:
    __device__ PoseidonPermutationGPU();
    
    __device__ void set_from_slice(gl64_t *elts, u32 len, u32 start_idx);

    __device__ void get_state_as_canonical_u64(u64* out);

    __device__ void set_state(u32 idx, gl64_t val);

    __device__ void permute();
    
    __device__ gl64_t *squeeze(u32 size);    
};

__device__ void poseidon_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash);

__device__ void poseidon_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);

#endif // __POSEIDON_V2_CUH__