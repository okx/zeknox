// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

namespace msm
{

#include <msm/msm.h>
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


}