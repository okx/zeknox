#ifndef __MONOLITH_CUH__
#define __MONOLITH_CUH__

#include "int_types.h"
#ifdef USE_CUDA
#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "poseidon.cuh"
#else
#include "goldilocks.hpp"
#include "poseidon.hpp"
#endif
#include "poseidon.h"
#include "monolith.h"

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