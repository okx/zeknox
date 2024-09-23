// Copyright 2024 OKX
#ifndef __MONOLITH_HPP__
#define __MONOLITH_HPP__

#include "types/int_types.h"
#ifdef USE_CUDA
#include "ff/gl64_t.cuh"
#include "utils/cuda_utils.cuh"
#include "poseidon/poseidon_permutation.cuh"
#else
#include "poseidon/goldilocks.hpp"
#include "poseidon/poseidon.hpp"
#include "poseidon/poseidon_permutation.hpp"
#include <cstring>
#endif

#include "merkle/hasher.hpp"

class MonolithHasher : public Hasher {

public:

#ifdef USE_CUDA
__host__ static void cpu_hash_one(u64 *input, u64 size, u64 *output);
__host__ static void cpu_hash_two(u64 *input1, u64 *input2, u64 *output);
__device__ static void gpu_hash_one(gl64_t *input, u32 size, gl64_t *output);
__device__ static void gpu_hash_two(gl64_t *input1, gl64_t *input2, gl64_t *output);
#else
static void cpu_hash_one(u64 *input, u64 size, u64 *output);
static void cpu_hash_two(u64 *input1, u64 *input2, u64 *output);
#endif

};

#ifdef USE_CUDA
class MonolithPermutationGPU : public PoseidonPermutationGPU
{
public:
    DEVICE void permute();
};
#else  // USE_CUDA
class MonolithPermutation : public PoseidonPermutation
{
public:
    void permute();
};
#endif // USE_CUDA

#endif // __MONOLITH_HPP__
