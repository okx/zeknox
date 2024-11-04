// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ZEKNOX_NATIVE_MSM_MSM_CU_
#define ZEKNOX_NATIVE_MSM_MSM_CU_

#include <msm/msm.h>
#include <utils/gpu_t.cuh>
#include <utils/all_gpus.hpp>
#include <msm/pippenger.cuh>
// #include <cstdio>

#ifdef FEATURE_BN254

RustError mult_pippenger_g1(uint32_t device_id,
                            void *result,
                            void *input_points,
                            void *input_scalars,
                            MSM_Config cfg)
{
    // printf("mult_pippenger_g1,npoints:%d\n", cfg.npoints);
    auto &gpu = select_gpu(device_id);
    gpu.select();

    dev_ptr_t<g1_affine_t> d_input_points{
        cfg.npoints,
        gpu,
        !cfg.are_inputs_on_device,
        cfg.are_inputs_on_device};

    dev_ptr_t<scalar_field_t> d_input_scalars{
        cfg.npoints,
        gpu,
        !cfg.are_inputs_on_device,
        cfg.are_inputs_on_device};

    if (cfg.are_inputs_on_device)
    {
        // printf("set input device pointer \n");
        d_input_points.set_device_ptr(reinterpret_cast<g1_affine_t *>(input_points));
        d_input_scalars.set_device_ptr(reinterpret_cast<scalar_field_t *>(input_scalars));
    }
    else
    {
        // printf("copy data from host to device \n");
        gpu.HtoD(&d_input_points[0], input_points, cfg.npoints);
        gpu.HtoD(&d_input_scalars[0], input_scalars, cfg.npoints);
    }

    g1_projective_t *result_jac = new g1_projective_t();

    RustError r = mult_pippenger_msm<g1_projective_t, g1_affine_t, scalar_field_t>(
        result_jac, &d_input_points[0], cfg.npoints, &d_input_scalars[0], cfg.are_input_point_in_mont, cfg.are_input_scalar_in_mont, cfg.are_outputs_on_device, cfg.big_triangle, cfg.large_bucket_factor);

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
// NOTE: G2 extension field compilation is very slow; only includes when necessary
// TODO: need to check why extension field compilation is so slow

// RustError mult_pippenger_g2(g2_projective_t *result, g2_affine_t *input_points, size_t msm_size, scalar_field_t *input_scalars, size_t large_bucket_factor, bool on_device, bool big_triangle)
RustError mult_pippenger_g2(uint32_t device_id,
                            void *result,
                            void *input_points,
                            void *input_scalars,
                            MSM_Config cfg)
{
    // printf("mult_pippenger_g2, input_scalars: %p, input_points: %p, npoints:%d\n", input_scalars, input_points, cfg.npoints);
    auto &gpu = select_gpu(device_id);
    gpu.select();

    dev_ptr_t<g2_affine_t> d_input_points{
        cfg.npoints,
        gpu,
        !cfg.are_inputs_on_device,
        cfg.are_inputs_on_device};

    dev_ptr_t<scalar_field_t> d_input_scalars{
        cfg.npoints,
        gpu,
        !cfg.are_inputs_on_device,
        cfg.are_inputs_on_device};

    if (cfg.are_inputs_on_device)
    {
        // printf("set input device pointer \n");
        d_input_points.set_device_ptr(reinterpret_cast<g2_affine_t *>(input_points));
        d_input_scalars.set_device_ptr(reinterpret_cast<scalar_field_t *>(input_scalars));
    }
    else
    {
        // printf("copy data from host to device \n");
        gpu.HtoD(&d_input_points[0], input_points, cfg.npoints);
        gpu.HtoD(&d_input_scalars[0], input_scalars, cfg.npoints);
    }

    g2_projective_t *result_jac = new g2_projective_t();

    RustError r = mult_pippenger_msm<g2_projective_t, g2_affine_t, scalar_field_t>(
        result_jac,
        &d_input_points[0],
        cfg.npoints,
        &d_input_scalars[0],
        cfg.are_input_point_in_mont,
        cfg.are_input_scalar_in_mont,
        cfg.are_outputs_on_device, cfg.big_triangle, cfg.large_bucket_factor);

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
#endif // G2_ENABLED

#endif // FEATURE_BN254

#endif // ZEKNOX_NATIVE_MSM_MSM_CU_