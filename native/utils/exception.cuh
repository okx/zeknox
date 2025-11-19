// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_CUDA_UTIL_EXCEPTION_CUH__
#define __ZEKNOX_CUDA_UTIL_EXCEPTION_CUH__

#include "exception.hpp"
#include <iostream>

using cuda_error = zeknox_error;

#define CUDA_OK(expr) do {                                  \
    cudaError_t code = expr;                                \
    if (code != cudaSuccess) {                              \
        printf("error code%d, %s \n", code, __FILE__);         \
        auto file = std::strstr(__FILE__, "zeknox");        \
        auto str = fmt("%s@%s:%d failed: \"%s\"", #expr,    \
                       file ? file : __FILE__, __LINE__,    \
                       cudaGetErrorString(code));           \
        throw cuda_error{-code, str};                       \
    }                                                       \
} while(0)

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void inline checkLast(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

#endif // __ZEKNOX_CUDA_UTIL_EXCEPTION_CUH__