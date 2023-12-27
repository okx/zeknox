#ifndef __CRYPTO_ARITHMEETIC_GL64_CU__
#define __CRYPTO_ARITHMEETIC_CL64_CU__
#include <arithmetic/kernels/gl64_kernel.cu>
#ifndef __CUDA_ARCH__   // below is cpu code; __CUDA_ARCH__ should not be defined

#if defined(EXPOSE_C_INTERFACE)
extern "C" 
#endif 
void goldilocks_add(fr_t *result, fr_t *a, fr_t *b)
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

#if defined(EXPOSE_C_INTERFACE)
extern "C" 
#endif 
void goldilocks_sub(fr_t *result, fr_t *a, fr_t *b)
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

#if defined(EXPOSE_C_INTERFACE)
extern "C" 
#endif 
void goldilocks_mul(fr_t *result, fr_t *a, fr_t *b)
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

#if defined(EXPOSE_C_INTERFACE)
extern "C" 
#endif 
void goldilocks_rshift(fr_t *result, fr_t *a, uint32_t *r)
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


#if defined(EXPOSE_C_INTERFACE)
extern "C" 
#endif 
void goldilocks_inverse(fr_t *result, fr_t *a)
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

#if defined(EXPOSE_C_INTERFACE)
extern "C" 
#endif 
void goldilocks_exp(fr_t *result, fr_t *base, uint32_t *pow)
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
#endif
#endif