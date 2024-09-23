// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _KECCAK_HPP_
#define _KECCAK_HPP_

#include "merkle/hasher.hpp"

class KeccakHasher : public Hasher {

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

#endif  // _KECCAK_HPP_
