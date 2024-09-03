#ifndef __POSEIDON_V2_CUH__
#define __POSEIDON_V2_CUH__

#include "types/int_types.h"
#include "types/gl64_t.cuh"
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
extern __device__ u32 GPU_MDS_MATRIX_CIRC[12];
extern __device__ u32 GPU_MDS_MATRIX_DIAG[12];
extern __device__ u64 GPU_ALL_ROUND_CONSTANTS[MAX_WIDTH * N_ROUNDS];

class PoseidonPermutationGPU : public PoseidonPermutationGPUVirtual
{
private:
    DEVICE static gl64_t reduce128(u128 x);

    DEVICE static gl64_t reduce_u160(u128 n_lo, u32 n_hi);

    DEVICE static void add_u160_u128(u128 *x_lo, u32 *x_hi, u128 y);

    DEVICE static gl64_t from_noncanonical_u96(gl64_t n_lo, gl64_t n_hi);

    DEVICE static gl64_t from_noncanonical_u128(u128 n);

    DEVICE void mds_partial_layer_fast(gl64_t *state, u32 r);

    DEVICE gl64_t mds_row_shf(u32 r, gl64_t *v);

    DEVICE void mds_layer(gl64_t *state, gl64_t *result);

    DEVICE void constant_layer(gl64_t *state, u32 *round_ctr);

    DEVICE static gl64_t sbox_monomial(const gl64_t &x);

    DEVICE void sbox_layer(gl64_t *state);

    DEVICE void full_rounds(gl64_t *state, u32 *round_ctr);

    DEVICE void partial_rounds_naive(gl64_t *state, u32 *round_ctr);

    DEVICE void partial_rounds(gl64_t *state, u32 *round_ctr);

    DEVICE gl64_t *poseidon_naive(gl64_t *input);

    DEVICE gl64_t *poseidon(gl64_t *input);

protected:
    gl64_t state[SPONGE_WIDTH];

public:
    DEVICE PoseidonPermutationGPU();

    DEVICE void set_from_slice(gl64_t *elts, u32 len, u32 start_idx);

    DEVICE void set_from_slice_stride(gl64_t *elts, u32 len, u32 start_idx, u32 stride);

    DEVICE void get_state_as_canonical_u64(u64* out);

    DEVICE void set_state(u32 idx, gl64_t val);

    DEVICE void permute();

    DEVICE gl64_t *squeeze(u32 size);
};

#ifdef DEBUG
DEVICE void print_perm(gl64_t *data, int cnt);
#endif

#endif  // USE_CUDA

#endif // __POSEIDON_V2_CUH__
