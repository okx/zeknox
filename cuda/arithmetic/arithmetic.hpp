// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO_ARITHMATIC_HPP__
#define __CRYPTO_ARITHMATIC_HPP__

#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif

#if defined(FEATURE_GOLDILOCKS)
#include <arithmetic/gl64.cu>
#elif defined(FEATURE_BN128)
#include <ff/alt_bn128.hpp>
__global__ void bn128_add_kernel(
    fp_t *d_result, fp_t *d_a, fp_t *d_b
    ) {

    *d_result = *d_a + *d_b;
}
#endif
#endif