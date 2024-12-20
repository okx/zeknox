// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_TYPES_H__
#define __ZEKNOX_TYPES_H__

#include <stdint.h>
#include <assert.h>

typedef uint8_t u8;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
#ifdef USE_CUDA
#include "cuda_uint128.cuh"
typedef uint128_t u128;
#else
typedef __uint128_t u128;
typedef __uint128_t uint128_t;
#endif

#endif // __ZEKNOX_TYPES_H__
