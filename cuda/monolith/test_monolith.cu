#include "monolith.cuh"
#include "poseidon.h"

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

    gpu_monolith_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void hash_step1(uint64_t *in, uint64_t *out, uint32_t n, uint32_t len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_monolith_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void hash_step2(uint64_t *in, uint64_t *out, uint32_t len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_monolith_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

int test1()
{
    u64 inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    u64 exp[4] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};

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

    assert(h1[0] == exp[0]);
    assert(h1[1] == exp[1]);
    assert(h1[2] == exp[2]);
    assert(h1[3] == exp[3]);

    cpu_monolith_hash_one(inp, 12, h2);
    printhash(h2);

    assert(h2[0] == exp[0]);
    assert(h2[1] == exp[1]);
    assert(h2[2] == exp[2]);
    assert(h2[3] == exp[3]);

    printf("Test 1: Results are ok!\n");

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
        cpu_monolith_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    cpu_monolith_hash_two(tree1, tree1 + 4, tree1 + 16);
    cpu_monolith_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    cpu_monolith_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

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
        printf("Test 2: Trees are the same!\n");
    }

    return ret;
}

int test3() {
    uint64_t inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    uint64_t hash[4] = {0};
    cpu_monolith_hash_one(inp, 12, hash);

    printf("Hash: \n");
    for(int i = 0; i < 4; i++) {
        printf("%lX ", hash[i]);
    }
    printf("\n");

    uint64_t ref[4] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};
    for(int i = 0; i < 4; i++) {
        assert(hash[i] == ref[i]);
    }

    printf("Test 3: Results are ok!\n");

    return 1;
}

int main()
{
    test1();

    test2();

    test3();

    return 0;
}