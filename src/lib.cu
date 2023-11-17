#include <cuda_runtime.h>
#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif
#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#include <ff/arithmatic.cuh>
#else
#error "no FEATURE"
#endif
#include <util/cuda_available.hpp>

#ifndef __CUDA_ARCH__
extern "C" void goldilocks_add(fr_t *result, fr_t *a, fr_t *b)
{

    printf("a val: %lu, b val: %lu \n", *a, *b);
    fr_t *d_result, *d_a, *d_b;
    cudaMalloc((fr_t**)&d_result, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_a, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_b, sizeof(fr_t));

    cudaMemcpy(d_a, a, sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(fr_t), cudaMemcpyHostToDevice);
    goldilocks_add_kernel<<<1,1>>>(
        d_result, d_a, d_b
        );

    cudaMemcpy(result, d_result, sizeof(fr_t), cudaMemcpyDeviceToHost);

}

extern "C" void mul(uint32_t *result)
{
    uint32_t *d_result;
    cudaMalloc((uint32_t **)&d_result, sizeof(uint32_t));
    mul_kernel<<<1, 1>>>(d_result);
    cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

#endif
