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
  if (tid < n_scalars * batch_size)
  {
    int scalar_id = tid % n_scalars;
   
    element_vec[tid] = scalar_vec[scalar_id] * element_vec[tid];
    //  printf("tid: %d, scalar: %lu, element: %lu, \n",tid, scalar_vec[scalar_id], element_vec[tid]);
  }
}
#endif