// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ZEKNOX_CUDA_LIB_H_
#define ZEKNOX_CUDA_LIB_H_

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

#include <utils/rusterror.h>
#include <ntt/ntt.h>
#include <merkle/merkle.h>

EXTERN RustError get_number_of_gpus(size_t *ngpus);

EXTERN RustError list_devices_info();

EXTERN void init_cuda();

EXTERN void init_cuda_degree(const uint32_t max_degree);

EXTERN RustError init_twiddle_factors(size_t device_id, size_t lg_n);

EXTERN RustError init_coset(size_t device_id, size_t lg_domain_size, const uint64_t coset_gen);

EXTERN RustError compute_batched_ntt(size_t device_id, void *inout, uint32_t lg_domain_size,
                                     NTT_Direction ntt_direction,
                                     NTT_Config cfg);

EXTERN RustError compute_batched_lde(size_t device_id, void *output, void *input, uint32_t lg_domain_size,
                                     NTT_Direction ntt_direction,
                                     NTT_Config cfg);

EXTERN RustError compute_batched_lde_multi_gpu(void *output, void *input, uint32_t num_gpu, NTT_Direction ntt_direction,
                                               NTT_Config cfg,
                                               uint32_t lg_domain_size,
                                               size_t total_num_input_elements,
                                               size_t total_num_output_elements);

EXTERN RustError compute_transpose_rev(size_t device_id, void *output, void *input, uint32_t lg_n,
                                       NTT_TransposeConfig cfg);

EXTERN RustError compute_naive_transpose_rev(size_t device_id, void *output, void *input, uint32_t lg_n,
                                             NTT_TransposeConfig cfg);

#ifdef FEATURE_BN254
#include <ff/alt_bn254.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
RustError::by_value mult_pippenger(point_t *out, const affine_t points[],
                                   size_t npoints, const scalar_t scalars[],
                                   size_t ffi_affine_sz);

#include <primitives/field.cuh>
#include <primitives/extension_field.cuh>
#include <primitives/projective.cuh>
#include <ff/bn254_params.cuh>
typedef Field<PARAMS_BN254::fp_config> scalar_field_t;
typedef Field<PARAMS_BN254::fq_config> point_field_t;
typedef ExtensionField<PARAMS_BN254::fq_config> g2_point_field_t;
static constexpr g2_point_field_t g2_gen_x =
    g2_point_field_t{point_field_t{PARAMS_BN254::g2_gen_x_re}, point_field_t{PARAMS_BN254::g2_gen_x_im}};
static constexpr g2_point_field_t g2_gen_y =
    g2_point_field_t{point_field_t{PARAMS_BN254::g2_gen_y_re}, point_field_t{PARAMS_BN254::g2_gen_y_im}};
static constexpr g2_point_field_t g2_b = g2_point_field_t{
    point_field_t{PARAMS_BN254::weierstrass_b_g2_re}, point_field_t{PARAMS_BN254::weierstrass_b_g2_im}};
typedef Projective<g2_point_field_t, scalar_field_t, g2_b, g2_gen_x, g2_gen_y> g2_projective_t;
typedef Affine<g2_point_field_t> g2_affine_t;

extern "C" RustError::by_value mult_pippenger_g2(g2_projective_t *out, g2_affine_t *points, size_t msm_size, scalar_field_t *scalars, size_t large_bucket_factor, bool on_device,
                                                 bool big_triangle);
#endif // FEATURE_BN254

#endif // ZEKNOX_CUDA_LIB_H_
