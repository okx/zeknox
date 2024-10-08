// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


#include <msm/msm.h>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <utils/gpu_t.cuh>
#include <utils/all_gpus.cpp>
#include <msm/pippenger.cuh>
#include <cstdio>
#include <blst_t.hpp>

RustError::by_value mult_pippenger(size_t device_id, point_t *out, affine_t *input_points, scalar_t *input_scalars, MSM_Config cfg)
{
    auto &gpu = select_gpu(device_id);
    gpu.select();

    dev_ptr_t<affine_t> d_input_points{
        cfg.npoints,
        gpu,
        !cfg.are_inputs_on_device,
        cfg.are_inputs_on_device
    };

    dev_ptr_t<scalar_t> d_input_scalars{
        cfg.npoints,
        gpu,
        !cfg.are_inputs_on_device,
        cfg.are_inputs_on_device
    };

    if (cfg.are_inputs_on_device)
    {
        // printf("set input device pointer \n");
        d_input_points.set_device_ptr(input_points);
        d_input_scalars.set_device_ptr(input_scalars);
    }
    else
    {
        // printf("copy data from host to device \n");
        gpu.HtoD(&d_input_points[0], input_points, cfg.npoints);
        gpu.HtoD(&d_input_scalars[0], input_scalars, cfg.npoints);
    }
    RustError r = mult_pippenger<bucket_t>(out, &d_input_points[0], cfg.npoints, &d_input_scalars[0], cfg.are_points_in_mont, cfg.ffi_affine_sz);
    return r;
}

#if defined(G2_ENABLED)
RustError::by_value mult_pippenger_g2(g2_projective_t *result, g2_affine_t *input_points, size_t msm_size, scalar_field_t *input_scalars, size_t large_bucket_factor, bool on_device, bool big_triangle)
{
    mult_pippenger_g2_internal<scalar_field_t, g2_projective_t, g2_affine_t>(
        result, input_points, input_scalars, msm_size, on_device, big_triangle, large_bucket_factor);
    CHECK_LAST_CUDA_ERROR();
}
#endif