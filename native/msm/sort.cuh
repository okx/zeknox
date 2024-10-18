// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_SORT_CUH__
#define __SPPARK_MSM_SORT_CUH__

/*
 * Custom sorting, we take in digits and return their indices.
 */
#include <cstdio>
#include <cstdint>
#include <utils/vec2d_t.hpp>

#define SORT_BLOCKDIM 1024
#ifndef DIGIT_BITS
# define DIGIT_BITS 13
#endif
#if DIGIT_BITS < 10 || DIGIT_BITS > 14
# error "impossible DIGIT_BITS"
#endif

#ifndef __MSM_SORT_DONT_IMPLEMENT__

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

static const uint32_t N_COUNTERS = 1<<DIGIT_BITS;
static const uint32_t N_SUMS = N_COUNTERS / SORT_BLOCKDIM;
extern __shared__ uint32_t counters[/*N_COUNTERS*/];

__global__ void sort(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1);

# undef asm
#endif
#endif
