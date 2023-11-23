#include <cuda.h>
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
#include <ntt/ntt.cuh>

#ifndef __CUDA_ARCH__   // below is cpu code; __CUDA_ARCH__ should not be defined

extern "C" void goldilocks_add(fr_t *result, fr_t *a, fr_t *b)
{

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

extern "C" void goldilocks_sub(fr_t *result, fr_t *a, fr_t *b)
{

    fr_t *d_result, *d_a, *d_b;
    cudaMalloc((fr_t**)&d_result, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_a, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_b, sizeof(fr_t));

    cudaMemcpy(d_a, a, sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(fr_t), cudaMemcpyHostToDevice);
    goldilocks_sub_kernel<<<1,1>>>(
        d_result, d_a, d_b
        );

    cudaMemcpy(result, d_result, sizeof(fr_t), cudaMemcpyDeviceToHost);

}

extern "C" void goldilocks_mul(fr_t *result, fr_t *a, fr_t *b)
{
       fr_t *d_result, *d_a, *d_b;
    cudaMalloc((fr_t**)&d_result, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_a, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_b, sizeof(fr_t));

    cudaMemcpy(d_a, a, sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(fr_t), cudaMemcpyHostToDevice);
    goldilocks_mul_kernel<<<1,1>>>(
        d_result, d_a, d_b
        );

    cudaMemcpy(result, d_result, sizeof(fr_t), cudaMemcpyDeviceToHost);
}

extern "C" void goldilocks_rshift(fr_t *result, fr_t *a, uint32_t *r)
{
       fr_t *d_result, *d_a;
       uint32_t *d_r;
    cudaMalloc((fr_t**)&d_result, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_a, sizeof(fr_t));
    cudaMalloc((uint32_t**)&d_r, sizeof(uint32_t));

    cudaMemcpy(d_a, a, sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sizeof(uint32_t), cudaMemcpyHostToDevice);
    goldilocks_rshift_kernel<<<1,1>>>(
        d_result, d_a, d_r
        );

    cudaMemcpy(result, d_result, sizeof(fr_t), cudaMemcpyDeviceToHost);
}


extern "C" void goldilocks_inverse(fr_t *result, fr_t *a)
{
       fr_t *d_result, *d_a;
    cudaMalloc((fr_t**)&d_result, sizeof(fr_t));
    cudaMalloc((fr_t**)&d_a, sizeof(fr_t));

    cudaMemcpy(d_a, a, sizeof(fr_t), cudaMemcpyHostToDevice);
    goldilocks_inverse_kernel<<<1,1>>>(
        d_result, d_a
        );

    cudaMemcpy(result, d_result, sizeof(fr_t), cudaMemcpyDeviceToHost);
}
#endif
