// Copyright 2024 OKX
#ifndef __CRYPTO_NTT_NTT_H__
#define __CRYPTO_NTT_NTT_H__
#include <map>
namespace Ntt_Types
{
    enum InputOutputOrder
    {
        NN,
        NR,
        RN,
        RR
    };
    enum Direction
    {
        forward,
        inverse
    };
    enum Type
    {
        standard,
        coset
    };
    enum Algorithm
    {
        GS,
        CT
    };

    struct NTTConfig
    {

        uint32_t batches;           /**< The number of NTTs to compute. Default value: 1. */
        InputOutputOrder order;     /**< Ordering of inputs and outputs. See [Ordering](@ref Ordering). Default value:
                                     *   `InputOutputOrder::NN`. */
        Type ntt_type;
        uint32_t extension_rate_bits;
        bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
        bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
        bool with_coset;
        bool is_multi_gpu;
        uint32_t salt_size;
    };

    struct TransposeConfig
    {
        uint32_t batches;           /**< The number of NTTs to compute. Default value: 1. */
        bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
        bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    };

}

#endif
