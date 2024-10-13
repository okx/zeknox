// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ZEKNOX_NATIVE_MSM_MSM_H_
#define ZEKNOX_NATIVE_MSM_MSM_H_

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
    _BOOL are_points_in_mont;   // whether input points are in montgomery form. Default value: false
    _BOOL are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    _BOOL are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */

    // below fields are MSM_G2 related
    uint32_t large_bucket_factor;
    _BOOL big_triangle;
} MSM_Config;

#ifdef FEATURE_BN254
#ifdef __cplusplus
#include <ff/alt_bn254.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
#endif
// EXTERN RustError::by_value mult_pippenger(size_t device_id, point_t* out, affine_t* points, scalar_t* scalars, MSM_Config cfg);
EXTERN RustError mult_pippenger(uint32_t device_id, void* out, void* points, void* scalars, MSM_Config cfg);

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
#endif // __cplusplus

// EXTERN RustError::by_value mult_pippenger_g2(g2_projective_t *out, g2_affine_t *points, size_t msm_size, scalar_field_t *scalars, size_t large_bucket_factor, bool on_device,
//                                              bool big_triangle);
EXTERN RustError mult_pippenger_g2(uint32_t device_id,
void *result,
 void *input_points,
 void *input_scalars,
// uint32_t npoints
MSM_Config cfg
);
#endif // FEATURE_BN254

#undef _BOOL

#endif