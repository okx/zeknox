#ifndef __POSEIDON2_CUH__
#define __POSEIDON2_CUH__

#include "types/int_types.h"
#ifdef USE_CUDA
#include "types/gl64_t.cuh"
#include "utils/cuda_utils.cuh"
#include "poseidon/poseidon.cuh"
#else
#include "poseidon/goldilocks.hpp"
#include "poseidon/poseidon.hpp"
#endif
#include "poseidon/poseidon.h"
#include "poseidon/poseidon2.h"

#ifdef USE_CUDA
class Poseidon2PermutationGPU : public PoseidonPermutationGPU
{
public:
    DEVICE void permute2();
};
#else  // USE_CUDA
class Poseidon2Permutation : public PoseidonPermutation
{
public:
    void permute2();
};
#endif // USE_CUDA

#endif // __POSEIDON2_CUH__