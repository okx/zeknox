#ifndef _POSEIDON_BN128_HPP_
#define _POSEIDON_BN128_HPP_

#include "hasher.hpp"

class PoseidonBN128Hasher : public Hasher {

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

#endif  // _POSEIDON_BN128_HPP_