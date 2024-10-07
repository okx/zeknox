// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/** macro utils*/
#define XSTR(x) STR(x)
#define STR(x) #x

/*
#ifdef NDEBUG
#define CUDA_DEBUG false
#else
#define CUDA_DEBUG true
#include <cstdio>
#endif
*/

// #pragma message "The value of CUDA_DEBUG: " XSTR(CUDA_DEBUG)
// #pragma message "The value of __CUDA_ARCH__: " XSTR(__CUDA_ARCH__)

#include <cuda.h>
#include <utils/gpu_t.cuh>
#include <utils/cuda_available.hpp>
#include <utils/all_gpus.cpp>
#include "lib.h"

#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn254.hpp>
#else
#error "no FEATURE"
#endif
#include <utils/device_context.cuh>

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    list_devices_info()
{
    list_all_gpus_prop();
    return RustError{cudaSuccess};
}

#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn254.hpp>
#else
#error "no FEATURE"
#endif

#include <ntt/ntt.cuh>
#include <ntt/ntt.h>
#include <vector>

#ifndef __CUDA_ARCH__ // below is cpu code; __CUDA_ARCH__ should not be defined

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    get_number_of_gpus(size_t *nums)
{
    *nums = ngpus();
    return RustError{cudaSuccess};
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    compute_batched_ntt(size_t device_id, void *inout, uint32_t lg_domain_size,
                        NTT_Direction ntt_direction, NTT_Config cfg)
{
    auto &gpu = select_gpu(device_id);
    return ntt::batch_ntt(gpu, (fr_t *)inout, lg_domain_size, ntt_direction, cfg);
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    compute_batched_lde(size_t device_id, void *output, void *input, uint32_t lg_domain_size,
                        NTT_Direction ntt_direction, NTT_Config cfg)
{
    auto &gpu = select_gpu(device_id);
    return ntt::batch_lde(gpu, (fr_t *)output, (fr_t *)input, lg_domain_size, ntt_direction, cfg);
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    compute_batched_lde_multi_gpu(void *output, void *input, uint32_t num_gpu, NTT_Direction ntt_direction,
                                  NTT_Config cfg, uint32_t lg_domain_size, size_t total_num_input_elements, size_t total_num_output_elements)
{
    return ntt::batch_lde_multi_gpu((fr_t *)output, (fr_t *)input, num_gpu, ntt_direction, cfg, lg_domain_size, total_num_input_elements, total_num_output_elements);
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    compute_transpose_rev(size_t device_id, void *output, void *input, uint32_t lg_n,
                          NTT_TransposeConfig cfg)
{
    auto &gpu = select_gpu(device_id);
    return ntt::compute_transpose_rev(gpu, (fr_t*)output, (fr_t*)input, lg_n, cfg);
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    compute_naive_transpose_rev(size_t device_id, void *output, void *input, uint32_t lg_n,
                                NTT_TransposeConfig cfg)
{
    auto &gpu = select_gpu(device_id);
    return ntt::compute_naive_transpose_rev(gpu, (fr_t*)output, (fr_t*)input, lg_n, cfg);
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    init_twiddle_factors(size_t device_id, size_t lg_n)
{
    auto &gpu = select_gpu(device_id);
    return ntt::init_twiddle_factors(gpu, lg_n);
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    init_coset(size_t device_id, size_t lg_n, const uint64_t coset_gen)
{
    auto &gpu = select_gpu(device_id);
    return ntt::init_coset(gpu, lg_n, fr_t(coset_gen));
}

#endif

#ifndef FEATURE_GOLDILOCKS
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <msm/pippenger.cuh>
#include <cstdio>
#include <blst_t.hpp>

RustError::by_value mult_pippenger(point_t *result, const affine_t points[],
                                   size_t npoints, const scalar_t scalars[],
                                   size_t ffi_affine_sz)
{
    RustError r = mult_pippenger<bucket_t>(result, points, npoints, scalars, false, ffi_affine_sz);
    return r;
}

#if defined(G2_ENABLED)
extern "C" RustError::by_value mult_pippenger_g2(g2_projective_t *result, g2_affine_t *points, size_t msm_size, scalar_field_t *scalars, size_t large_bucket_factor, bool on_device, bool big_triangle)
{
    mult_pippenger_g2_internal<scalar_field_t, g2_projective_t, g2_affine_t>(
        result, points, scalars, msm_size, on_device, big_triangle, large_bucket_factor);
    CHECK_LAST_CUDA_ERROR();
}
#endif
#endif

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    void
    init_cuda_degree_and_generator(const uint32_t max_degree, const uint64_t group_generator)
{
#if defined(FEATURE_GOLDILOCKS)
    size_t num_of_gpus = ngpus();

    for (size_t d = 0; d < num_of_gpus; d++)
    {
        init_coset(d, max_degree, group_generator);
        for (size_t k = 2; k <= max_degree; k++)
        {
            init_twiddle_factors(d, k);
        }
    }
#endif
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    void
    init_cuda_degree(const uint32_t max_degree)
{
#if defined(FEATURE_GOLDILOCKS)
    init_cuda_degree_and_generator(max_degree, GROUP_GENERATOR);
#endif
}

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    void
    init_cuda()
{
    init_cuda_degree(24);
}