#ifndef __CRYPTO_ARITHMATIC_CUH__
#define __CRYPTO_ARITHMATIC_CUH__

#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif

#include <ff/goldilocks.hpp>
__global__ void goldilocks_add_kernel(
    fr_t *d_result, fr_t *d_a, fr_t *d_b
    ) {

    printf("inside kernel new \n");
    printf("d_a: %lu , d_b: %lu, d_result: %lu \n", *d_a, *d_b, *d_result);
    *d_result = *d_a + *d_b;
   printf("d_a: %lu , d_b: %lu, d_result: %lu \n", *d_a, *d_b, *d_result);
}

__global__ void mul_kernel(uint32_t *result)
{
    uint32_t a = (1 << 31) +2;
    uint32_t b = 1 << 2;
    uint32_t lo, hi;
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(lo) ,"=r"(hi)  : "r"(a), "r"(b));
    printf("lo: %u , hi: %u \n", lo, hi);
    *result = lo;
}

#endif