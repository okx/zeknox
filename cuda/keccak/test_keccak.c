#include "int_types.h"

// #define RUST_POSEIDON

#include "keccak.h"

#include <stdio.h>

void printhash(u64 *h)
{
    for (int i = 0; i < 4; i++)
    {
        printf("%lu ", h[i]);
    }
    printf("\n");
}

#ifdef USE_CUDA
__global__ gpu_driver(uint64_t *input, uint32_t size, uint64_t *hash)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1)
        return;
    gpu_keccak_hash_one(input, size, hash);
}

void run_on_gpu(uint64_t *input, uint32_t size, uint64_t *hash)
{
    u64 *gpu_leaf, *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 6 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_leaf, leaf, 6 * sizeof(u64), cudaMemcpyHostToDevice));
    run_on_gpu<<<1, 1>>>(gpu_leaf, 1, gpu_hash);
    CHECKCUDAERR(cudaMemcpyAsync(gpu_hash, hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_leaf));
    CHECKCUDAERR(cudaFree(gpu_hash));
}
#endif

int main()
{

    u64 leaf[6] = {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 13588825262671526271ul, 13421290117754017454ul, 7401888676587830362ul};

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    for (int size = 1; size <= 6; size++)
    {
        printf("*** Size %d\n", size);
        cpu_keccak_hash_one(leaf, size, h1);
#ifdef USE_CUDA
        run_on_gpu(leaf, size, h2);
#endif
        // ext_keccak_hash_or_noop(h1, leaf, size);
        printhash(h1);
        printhash(h2);
    }

    return 0;
}