#ifndef __MONOLITH_HPP__
#define __MONOLITH_HPP__

#include "int_types.h"
#ifdef USE_CUDA
#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "poseidon.cuh"
#else
#include "goldilocks.hpp"
#include "poseidon.hpp"
#endif

#include "hasher.hpp"

class MonolithHasher : public Hasher {

public:

#ifdef USE_CUDA
__host__ void cpu_hash_one(uint64_t *input, uint64_t size, uint64_t *output);
__host__ void cpu_hash_two(uint64_t *input1, uint64_t *input2, uint64_t *output);
__device__ void gpu_hash_one(gl64_t *input, uint32_t size, gl64_t *output);
__device__ void gpu_hash_two(gl64_t *input1, gl64_t *input2, gl64_t *output);
#else
void cpu_hash_one(uint64_t *input, uint64_t size, uint64_t *output);
void cpu_hash_two(uint64_t *input1, uint64_t *input2, uint64_t *output);
#endif

};

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

#endif // __MONOLITH_HPP__