#ifndef __POSEIDON2_CUH__
#define __POSEIDON2_CUH__

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
#include "poseidon2.h"

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