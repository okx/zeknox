#ifndef UTILS_KERNELS_H
#define UTILS_KERNELS_H

#include <util/batch_mul.cu>
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

#endif

#endif