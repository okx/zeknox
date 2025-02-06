// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_CUDA_NTT_NTT_H__
#define __ZEKNOX_CUDA_NTT_NTT_H__

#include <stdint.h>

// This is because Rust uses a C++ binding, while Go uses a C binding.
#ifdef __cplusplus
#define _BOOL bool
#else
#define _BOOL char
#endif

typedef enum
{
    NN,
    NR,
    RN,
    RR
} NTT_InputOutputOrder;

typedef enum
{
    forward,
    inverse
} NTT_Direction;

typedef enum
{
    standard,
    coset
} NTT_Type;

typedef enum
{
    GS,
    CT
} NTT_Algorithm;

typedef struct
{
    uint32_t batches;           /**< The number of NTTs to compute. Default value: 1. */
    NTT_InputOutputOrder order; /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
                                 *   `InputOutputOrder::NN`. */
    NTT_Type ntt_type;
    uint32_t extension_rate_bits;
    _BOOL are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    _BOOL are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    _BOOL with_coset;
    _BOOL is_multi_gpu;
    uint32_t salt_size;
} NTT_Config;

typedef struct
{
    uint32_t batches;          /**< The number of NTTs to compute. Default value: 1. */
    _BOOL are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    _BOOL are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
} NTT_TransposeConfig;

#undef _BOOL

#endif // __ZEKNOX_CUDA_NTT_NTT_H__
