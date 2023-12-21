#ifndef __CRYPTO_ARITHMEETIC_GL64_CU__
#define __CRYPTO_ARITHMEETIC_CL64_CU__
#include <arithmetic/kernels/bn128_kernel.cu>
#ifndef __CUDA_ARCH__   // below is cpu code; __CUDA_ARCH__ should not be defined

extern "C" void bn128_add(fp_t *result, fp_t *a, fp_t *b)
{

    fp_t *d_result, *d_a, *d_b;
    cudaMalloc((fp_t**)&d_result, sizeof(fp_t));
    cudaMalloc((fp_t**)&d_a, sizeof(fp_t));
    cudaMalloc((fp_t**)&d_b, sizeof(fp_t));

    cudaMemcpy(d_a, a, sizeof(fp_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(fp_t), cudaMemcpyHostToDevice);
    bn128_add_kernel<<<1,1>>>(
        d_result, d_a, d_b
        );

    cudaMemcpy(result, d_result, sizeof(fp_t), cudaMemcpyDeviceToHost);

}

#endif
#endif