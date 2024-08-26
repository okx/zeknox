#ifndef __POSEIDON_V2_CUH__
#define __POSEIDON_V2_CUH__

#include "int_types.h"
#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "poseidon.h"

#define MIN(x, y) (x < y) ? x : y

#define NUM_HASH_OUT_ELTS 4

#define SPONGE_RATE 8
#define SPONGE_CAPACITY 4
#define SPONGE_WIDTH 12

#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL 8
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS 30
#define MAX_WIDTH 12

// extern __device__ u32 GPU_MDS_MATRIX_CIRC[12];
// extern __device__ u32 GPU_MDS_MATRIX_DIAG[12];
extern __device__ u64 GPU_ALL_ROUND_CONSTANTS[MAX_WIDTH * N_ROUNDS];

#ifdef USE_CUDA
class PoseidonPermutationGPU
#else
class PoseidonPermutation
#endif
{
private:
    DEVICE static gl64_t reduce128(uint128_t x);

    DEVICE static gl64_t reduce_u160(uint128_t n_lo, uint32_t n_hi);

    DEVICE static void add_u160_u128(u128 *x_lo, u32 *x_hi, u128 y);

    DEVICE static gl64_t from_noncanonical_u96(gl64_t n_lo, gl64_t n_hi);

    DEVICE static gl64_t from_noncanonical_u128(u128 n);

    DEVICE void mds_partial_layer_fast(gl64_t *state, u32 r);

    DEVICE gl64_t mds_row_shf(u32 r, gl64_t *v);

    DEVICE void mds_layer(gl64_t *state, gl64_t *result);

    DEVICE void constant_layer(gl64_t *state, gl64_t* rconst, u32 *round_ctr);

    DEVICE static gl64_t sbox_monomial(const gl64_t &x);

    DEVICE void sbox_layer(gl64_t *state);

    DEVICE void full_rounds(gl64_t *state, gl64_t* rconst, u32 *round_ctr);

    DEVICE void partial_rounds_naive(gl64_t *state, gl64_t* rconst, u32 *round_ctr);

    DEVICE void partial_rounds(gl64_t *state, u32 *round_ctr);

    DEVICE gl64_t *poseidon_naive(gl64_t *input, gl64_t* rconst);

    DEVICE gl64_t *poseidon(gl64_t *input, gl64_t* rconst);

protected:
    gl64_t state[SPONGE_WIDTH];

public:
    DEVICE PoseidonPermutationGPU();

    DEVICE void set_from_slice(gl64_t *elts, u32 len, u32 start_idx);

    DEVICE void set_from_slice_stride(gl64_t *elts, u32 len, u32 start_idx, u32 stride);

    DEVICE void get_state_as_canonical_u64(u64* out);

    DEVICE void set_state(u32 idx, gl64_t val);

    DEVICE void permute(gl64_t* rconst);

    DEVICE gl64_t *squeeze(u32 size);
};

#ifdef DEBUG
DEVICE void print_perm(gl64_t *data, int cnt);
#endif

#endif // __POSEIDON_V2_CUH__