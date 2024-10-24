// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <msm/msm.h>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <utils/gpu_t.cuh>
#include <utils/all_gpus.hpp>
#include <msm/pippenger.cuh>
#include <cstdio>

RustError mult_pippenger_g1(uint32_t device_id,
                            void *result,
                            void *input_points,
                            void *input_scalars,
                            MSM_Config cfg)
{
    // printf("mult_pippenger_g1 \n");
    // printf("start mult_pippenger_g2, device_id: %d, npoints: %d, mont: %d, are_inputs_on_device: %d, big_triangle:%d, large_bucket_factor: %d \n",
    // device_id, cfg.npoints, cfg.are_points_in_mont, cfg.are_inputs_on_device, cfg.big_triangle, cfg.large_bucket_factor);

    scalar_field_t *scalar = reinterpret_cast<scalar_field_t *>(input_scalars);
    g1_affine_t *points = reinterpret_cast<g1_affine_t *>(input_points);

    g1_projective_t *result_jac = new g1_projective_t();

    RustError r = mult_pippenger_internal<scalar_field_t, g1_projective_t, g1_affine_t>(
        result_jac, points, scalar, cfg.npoints, cfg.are_input_point_in_mont,cfg.are_input_scalar_in_mont, cfg.are_outputs_on_device, cfg.big_triangle, cfg.large_bucket_factor);

    g1_affine_t gpu_result_affine = g1_projective_t::to_affine(*result_jac);
    if (cfg.are_output_point_in_mont)
    {
        g1_affine_t gpu_result_affine_mont = g1_affine_t::to_montgomery(gpu_result_affine);
        *reinterpret_cast<g1_affine_t *>(result) = gpu_result_affine_mont;
    }
    else
    {
        *reinterpret_cast<g1_affine_t *>(result) = gpu_result_affine;
    }

    return r;
}

#if defined(G2_ENABLED)
// RustError mult_pippenger_g2(g2_projective_t *result, g2_affine_t *input_points, size_t msm_size, scalar_field_t *input_scalars, size_t large_bucket_factor, bool on_device, bool big_triangle)
RustError mult_pippenger_g2(uint32_t device_id,
                            void *result,
                            void *input_points,
                            void *input_scalars,
                            MSM_Config cfg)
{
    // printf("start mult_pippenger_g2, device_id: %d, npoints: %d, mont: %d, are_inputs_on_device: %d, big_triangle:%d, large_bucket_factor: %d \n",
    // device_id, cfg.npoints, cfg.are_points_in_mont, cfg.are_inputs_on_device, cfg.big_triangle, cfg.large_bucket_factor);

    scalar_field_t *scalar = reinterpret_cast<scalar_field_t *>(input_scalars);
    g2_affine_t *points = reinterpret_cast<g2_affine_t *>(input_points);

    g2_projective_t *result_jac = new g2_projective_t();

    RustError r = mult_pippenger_internal<scalar_field_t, g2_projective_t, g2_affine_t>(
        result_jac, points, scalar, cfg.npoints, cfg.are_input_point_in_mont,cfg.are_input_scalar_in_mont, cfg.are_outputs_on_device, cfg.big_triangle, cfg.large_bucket_factor);

    g2_affine_t gpu_result_affine = g2_projective_t::to_affine(*result_jac);
    if (cfg.are_output_point_in_mont)
    {
        g2_affine_t gpu_result_affine_mont = g2_affine_t::to_montgomery(gpu_result_affine);
        *reinterpret_cast<g2_affine_t *>(result) = gpu_result_affine_mont;
    }
    else
    {
        *reinterpret_cast<g2_affine_t *>(result) = gpu_result_affine;
    }

    return r;
}
#endif
