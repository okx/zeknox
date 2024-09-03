#ifndef __POSEIDON2_HPP__
#define __POSEIDON2_HPP__

#include "types/int_types.h"
#ifdef USE_CUDA
#include "types/gl64_t.cuh"
#include "utils/cuda_utils.cuh"
#include "poseidon/poseidon.cuh"
#else
#include "poseidon/goldilocks.hpp"
#include "poseidon/poseidon.hpp"
#endif

#include "merkle/hasher.hpp"

class Poseidon2Hasher : public Hasher {

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

#endif // __POSEIDON2_HPP__