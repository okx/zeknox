// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __UTILS_BATCH_MUL_CU__
#define __UTILS_BATCH_MUL_CU__

#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn254.hpp>
#else
#error "no FEATURE"
#endif

__global__ void batchVectorMult(fr_t *scalar_vec, fr_t *element_vec, unsigned n_scalars, unsigned batch_size)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // printf("tid: %d, element0: %lu, elementn: %lu,  elementn+1: %lu, \n", tid, scalar_vec[0], scalar_vec[(1<<17) - 1],  scalar_vec[1<<17]);
  if (tid < n_scalars * batch_size)
  {
    // printf("inside kernel batchVectorMult, tid: %d, n_scalars: %d, batch_size: %d \n", tid, n_scalars, batch_size);
    int scalar_id = tid % n_scalars;
    // printf("tid: %d, scalar: %lu, element: %lu, \n", tid, scalar_vec[scalar_id], element_vec[tid]);
    element_vec[tid] = scalar_vec[scalar_id] * element_vec[tid];
    //
  }
}
#endif