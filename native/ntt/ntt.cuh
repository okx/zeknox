// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_NTT_NTT_CUH__
#define __ZEKNOX_NTT_NTT_CUH__

#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn254.hpp>
#else
#error "no FEATURE"
#endif

#include <utils/gpu_t.cuh>
#include <utils/rusterror.h>
#include <ntt/ntt.h>

namespace ntt {
RustError init_twiddle_factors(const gpu_t &gpu, size_t lg_domain_size);

RustError init_coset(const gpu_t &gpu, size_t lg_domain_size, fr_t coset_gen);

RustError batch_ntt(const gpu_t &gpu, fr_t *inout, uint32_t lg_domain_size, NTT_Direction direction, NTT_Config cfg);

RustError batch_lde(const gpu_t &gpu, fr_t *output, fr_t *input, uint32_t lg_n, NTT_Direction direction, NTT_Config cfg);

RustError batch_lde_multi_gpu(fr_t *output, fr_t *inputs, size_t num_gpu, NTT_Direction direction, NTT_Config cfg, size_t lg_n, size_t total_num_input_elements, size_t total_num_output_elements);

RustError compute_transpose_rev(const gpu_t &gpu, fr_t *output, fr_t *input, uint32_t lg_n, NTT_TransposeConfig cfg);
}

#endif  // __ZEKNOX_NTT_NTT_CUH__