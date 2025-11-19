// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_UTILS_BATCH_MUL_CUH__
#define __ZEKNOX_UTILS_BATCH_MUL_CUH__

#include <utils/batch_mul.cu>
#define MAX_THREADS_PER_BLOCK 512

#ifndef __CUDA_ARCH__
int batch_vector_mult(fr_t *scalar_vec, fr_t *element_vec, unsigned n_scalars, unsigned batch_size, cudaStream_t stream)
{

  int NUM_THREADS = MAX_THREADS_PER_BLOCK;
  int NUM_BLOCKS = (n_scalars * batch_size + NUM_THREADS - 1) / NUM_THREADS;
  // printf("mul coset, NUM_BLOCKS:%d, NUM_THREADS: %d, n_scalars:%d \n", NUM_BLOCKS, NUM_THREADS, n_scalars);
  // TODO: do no have to mul for zeros elements
  batchVectorMult<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(scalar_vec, element_vec, n_scalars, batch_size);
  return 0;
}

#endif // __CUDA_ARCH__

#endif // __ZEKNOX_UTILS_BATCH_MUL_CUH__