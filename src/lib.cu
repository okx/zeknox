#include <cuda.h>
#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif
#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#include <ff/arithmatic.cuh>
#elif defined(FEATURE_BN128)
# include <ff/alt_bn128.hpp>
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

extern "C" void goldilocks_exp(fr_t *result, fr_t *base, uint32_t *pow)
{
    fr_t *d_result, *d_base;
    uint32_t *d_pow;
    cudaMalloc((fr_t **)&d_result, sizeof(fr_t));
    cudaMalloc((fr_t **)&d_base, sizeof(fr_t));
    cudaMalloc((uint32_t **)&d_pow, sizeof(uint32_t));

    cudaMemcpy(d_base, base, sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pow, pow, sizeof(uint32_t), cudaMemcpyHostToDevice);
    goldilocks_exp_kernel<<<1, 1>>>(
        d_result, d_base, d_pow);

    cudaMemcpy(result, d_result, sizeof(fr_t), cudaMemcpyDeviceToHost);
}
extern "C" RustError compute_ntt(size_t device_id, fr_t *inout, uint32_t lg_domain_size,
                                 NTT::InputOutputOrder ntt_order,
                                 NTT::Direction ntt_direction,
                                 NTT::Type ntt_type)
{
    auto &gpu = select_gpu(device_id);

    return NTT::Base(gpu, inout, lg_domain_size,
                     ntt_order, ntt_direction, ntt_type);
}
#endif
