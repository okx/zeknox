#include "keccak.hip"

#include "keccak_cpu.h"

#include <stdio.h>

void printhash(uint64_t *h)
{
    for (int i = 0; i < 4; i++)
    {
        printf("%lu ", h[i]);
    }
    printf("\n");
}

__global__ void gpu_driver(uint64_t *input, uint32_t size, uint64_t *hash)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1)
        return;
    // gpu_keccak_hash_one(input, size, hash);
    keccak((uint8_t *)input, size, (uint8_t *)hash, 32);
}

void run_on_gpu(uint64_t *input, uint32_t size, uint64_t *hash)
{
    uint64_t *gpu_leaf, *gpu_hash;
    HIP_CHECK(hipMalloc(&gpu_leaf, 6 * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&gpu_hash, 4 * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(gpu_leaf, input, 6 * sizeof(uint64_t), hipMemcpyHostToDevice));
    gpu_driver<<<1, 32>>>(gpu_leaf, size, gpu_hash);
    HIP_CHECK(hipMemcpy(hash, gpu_hash, 4 * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(gpu_leaf));
    HIP_CHECK(hipFree(gpu_hash));
}

int main()
{

    uint64_t leaf[6] = {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 13588825262671526271ul, 13421290117754017454ul, 7401888676587830362ul};

    uint64_t h1[4] = {0u};
    uint64_t h2[4] = {0u};

    for (int size = 1; size <= 6; size++)
    {
        printf("*** Size %d\n", size);
        keccak((uint8_t *)leaf, size, (uint8_t *)h1, 32);
        run_on_gpu(leaf, size, h2);
        printhash(h1);
        printhash(h2);
        for (int j = 0; j < 4; j++)
            assert(h1[j] == h2[j]);
    }

    printf("Test done ok!\n");
    return 0;
}