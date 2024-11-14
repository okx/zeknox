// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_NATIVE_MSM_MSM_H__
#define __ZEKNOX_NATIVE_MSM_MSM_H__

#include <stdint.h>
#include <utils/rusterror.h>

#ifdef __cplusplus
#define _BOOL bool
#define EXTERN extern "C"
#else
#define _BOOL char
#define EXTERN
#endif

typedef struct
{
    uint32_t ffi_affine_sz;       // affine point size; for bn254 is 64; 32 bytes for X and 32 bytes for Y.
    uint32_t npoints;             // number of points
    _BOOL are_input_point_in_mont;   // whether input points are in montgomery form. Default value: false
    _BOOL are_input_scalar_in_mont;   // whether input scalar are in montgomery form. Default value: false
    _BOOL are_output_point_in_mont;   // whether output points are in montgomery form. Default value: false
    _BOOL are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    _BOOL are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */

    // below fields are MSM_G2 related
    uint32_t large_bucket_factor; // to control the threshhold of large bucket
    _BOOL big_triangle; // select the type of reduction kernel, default is not using big_triangle (with very poor performance)
} MSM_Config;

#ifdef FEATURE_BN254
#ifdef __cplusplus
#include <stdexcept>
#include <cstdint>
#include <iostream>
// NOTE: below definitions are for gl64 fields; most MSM is not performed on that field; can enable when necessary
// #include <blst_t.hpp>
// #include <ff/alt_bn254.hpp>
// #include <ec/jacobian_t.hpp>
// #include <ec/xyzz_t.hpp>
// typedef jacobian_t<fp_t> point_t;
// typedef xyzz_t<fp_t> bucket_t;
// typedef bucket_t::affine_t affine_t;
// typedef fr_t scalar_t;
#endif

#ifdef __cplusplus
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

typedef Affine<point_field_t> g1_affine_t;

static constexpr point_field_t g1_b = point_field_t{PARAMS_BN254::weierstrass_b};
static constexpr point_field_t g1_gen_x = point_field_t{PARAMS_BN254::g1_gen_x};
static constexpr point_field_t g1_gen_y = point_field_t{PARAMS_BN254::g1_gen_y};
typedef Projective<
    point_field_t,
    scalar_field_t,
    g1_b,
    g1_gen_x,
    g1_gen_y
> g1_projective_t;
#endif // __cplusplus

EXTERN RustError mult_pippenger_g1(uint32_t device_id,
void *result,
 void *input_points,
 void *input_scalars,
MSM_Config cfg
);


// EXTERN RustError::by_value mult_pippenger_g2(g2_projective_t *out, g2_affine_t *points, size_t msm_size, scalar_field_t *scalars, size_t large_bucket_factor, bool on_device,
//                                              bool big_triangle);
EXTERN RustError mult_pippenger_g2(uint32_t device_id,
void *result,
 void *input_points,
 void *input_scalars,
MSM_Config cfg
);

#endif // FEATURE_BN254

#undef _BOOL

#endif  // __ZEKNOX_NATIVE_MSM_MSM_H__