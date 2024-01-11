#ifndef __CRYPTO_ARITHMATIC_KERNELS_BN128_CU__
#define __CRYPTO_ARITHMATIC_KERNELS_BN128_CU__

#include <ff/alt_bn254.hpp>
__global__ void bn128_add_kernel(
    fp_t *d_result, fp_t *d_a, fp_t *d_b)
{
    *d_result = *d_a + *d_b;
}

__global__ void bn128_sub_kernel(
    fp_t *d_result, fp_t *d_a, fp_t *d_b)
{
    *d_result = *d_a - *d_b;
}

__global__ void bn128_mul_kernel(
    fp_t *d_result, fp_t *d_a, fp_t *d_b)
{
    *d_result = *d_a * (*d_b);
}

__global__ void bn128_lshift_kernel(fp_t *d_result, fp_t *d_a, uint32_t *l)
{
    *d_result = *d_a << (*l);
}

__global__ void bn128_rshift_kernel(fp_t *d_result, fp_t *d_a, uint32_t *r)
{
    *d_result = *d_a >> (*r);
}


#endif