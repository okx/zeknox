// Copyright 2024 OKX
#ifndef ZEKNOX_CUDA_NTT_NTT_CUH_
#define ZEKNOX_CUDA_NTT_NTT_CUH_

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

RustError batch_lde_multi_gpu(fr_t *output, fr_t *inputs, size_t num_gpu, NTT_Direction direction, NTT_Config cfg, size_t lg_n);

RustError transpose_and_bit_rev_batch(const gpu_t &gpu, fr_t *output, fr_t *input, uint32_t lg_n, NTT_TransposeConfig cfg);
}

#endif  // ZEKNOX_CUDA_NTT_NTT_CUH_