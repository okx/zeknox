#ifndef __MONOLITH_CUH__
#define __MONOLITH_CUH__

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
#include "monolith/monolith.h"

#ifdef USE_CUDA
class MonolithPermutationGPU : public PoseidonPermutationGPU
{
public:
    DEVICE void permute2();
};
#else  // USE_CUDA
class MonolithPermutation : public PoseidonPermutation
{
public:
    void permute2();
};
#endif // USE_CUDA

#endif // __MONOLITH_CUH__