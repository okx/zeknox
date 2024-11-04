// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO_UTIL_DEVICE_CONTEXT_CUH__
#define __CRYPTO_UTIL_DEVICE_CONTEXT_CUH__

#include <cuda_runtime.h>

namespace device_context {

  struct DeviceContext {
    cudaStream_t& stream;  /**< Stream to use. Default value: 0. */
    std::size_t device_id; /**< Index of the currently used GPU. Default value: 0. */
    cudaMemPool_t mempool; /**< Mempool to use. Default value: 0. */
  };

  /**
   * Return default device context that corresponds to using the default stream of the first GPU
   */
  inline DeviceContext get_default_device_context()
  {
    static cudaStream_t default_stream = (cudaStream_t)0;
    return DeviceContext{
      (cudaStream_t&)default_stream, // stream
      0,                             // device_id
      0,                             // mempool
    };
  }

} // namespace device_context

#endif