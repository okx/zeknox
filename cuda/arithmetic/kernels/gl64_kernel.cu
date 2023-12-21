#ifndef __CRYPTO_ARITHMATIC_KERNELS_GL64_CU__
#define __CRYPTO_ARITHMATIC_KERNELS_GL64_CU__

#include <ff/goldilocks.hpp>
__global__ void goldilocks_add_kernel(
    fr_t *d_result, fr_t *d_a, fr_t *d_b
    ) {

    *d_result = *d_a + *d_b;
}

__global__ void goldilocks_sub_kernel(
    fr_t *d_result, fr_t *d_a, fr_t *d_b
    ) {

    *d_result = *d_a - *d_b;
}

__global__ void goldilocks_mul_kernel(fr_t *d_result, fr_t *d_a, fr_t *d_b)
{
    *d_result = *d_a * *d_b;
}


__global__ void goldilocks_inverse_kernel(fr_t *d_result, fr_t *d_a)
{
    *d_result = 1/ *d_a;
}

__global__ void goldilocks_rshift_kernel(fr_t *d_result, fr_t *d_a, uint32_t *r)
{
    *d_result = *d_a>>(*r);
}

__global__ void goldilocks_exp_kernel(fr_t *d_result, fr_t *d_a, uint32_t *r)
{
    *d_result = (*d_a)^(*r);
}
#endif