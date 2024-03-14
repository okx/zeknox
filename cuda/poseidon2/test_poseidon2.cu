#include "poseidon2.cuh"
#include "poseidon2.h"

#include <stdio.h>

void printhash(u64 *h)
{
    for (int i = 0; i < 4; i++)
    {
        printf("%lx ", h[i]);
    }
    printf("\n");
}

__global__ void hash(uint64_t *in, uint64_t *out, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_poseidon2_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void hash_step1(uint64_t *in, uint64_t *out, uint32_t n, uint32_t len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_poseidon2_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void hash_step2(uint64_t *in, uint64_t *out, uint32_t len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_poseidon2_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

int test1()
{
    // similar to the test in goldilocks repo
    // test 1 - Fibonacci
    u64 inp[12];
    inp[0] = 0;
    inp[1] = 1;
    for (int i = 2; i < 12; i++)
    {
        inp[i] = inp[i-2] + inp[i-1];
    }

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    hash<<<1, 1>>>(gpu_leaf, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(h1, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
    printhash(h1);

    assert(h1[0] == 0x133a03eca11d93fb);
    assert(h1[1] == 0x5365414fb618f58d);
    assert(h1[2] == 0xfa49f50f3a2ba2e5);
    assert(h1[3] == 0xd16e53672c9832a4);

    cpu_poseidon2_hash_one(inp, 12, h2);
    printhash(h2);

    assert(h2[0] == 0x133a03eca11d93fb);
    assert(h2[1] == 0x5365414fb618f58d);
    assert(h2[2] == 0xfa49f50f3a2ba2e5);
    assert(h2[3] == 0xd16e53672c9832a4);

    return 1;
}

int test2()
{
    // 4 leaves of 7 elements each -> Merkle tree has 7 nodes
    u64 test_leaves[28] = {
        12382199520291307008, 18193113598248284716, 17339479877015319223, 10837159358996869336, 9988531527727040483, 5682487500867411209, 13124187887292514366,
        8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027,
        10465118329878758468, 5866464242232862106, 15506463679657361352, 18404485636523119190, 15311871720566825080, 5967980567132965479, 14180845406393061616,
        15480539652174185186, 5454640537573844893, 3664852224809466446, 5547792914986991141, 5885254103823722535, 6014567676786509263, 11767239063322171808};

    // CPU
    u64 tree1[28] = {0ul};

    for (u32 i = 0; i < 4; i++)
    {
        cpu_poseidon2_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    cpu_poseidon2_hash_two(tree1, tree1 + 4, tree1 + 16);
    cpu_poseidon2_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    cpu_poseidon2_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

    // GPU
    u64 tree2[28] = {0ul};

    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    hash_step1<<<1, 4>>>(gpu_leaf, gpu_hash, 7, 4);
    hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    int ret = 1;
    for (u32 i = 0; i < 28; i++)
    {
        if (tree1[i] != tree2[i])
        {
            printf("Diff at idx %u: %lu %lu\n", i, tree1[i], tree2[i]);
            ret = 0;
        }
    }
    if (ret == 1)
    {
        printf("Trees are the same!\n");
    }

    return ret;
}

int main()
{
    test1();

    test2();

    return 0;
}