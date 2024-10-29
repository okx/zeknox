// Copyright 2024 OKX
#ifndef ZEKNOX_CUDA_NTT_NTT_H_
#define ZEKNOX_CUDA_NTT_NTT_H_

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
    _BOOL is_coeffs;
    uint32_t salt_size;
} NTT_Config;

typedef struct
{
    uint32_t batches;          /**< The number of NTTs to compute. Default value: 1. */
    _BOOL are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    _BOOL are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
} NTT_TransposeConfig;

#undef _BOOL

#endif // ZEKNOX_CUDA_NTT_NTT_H_
