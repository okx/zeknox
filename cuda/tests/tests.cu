#include <gtest/gtest.h>
#include <assert.h>

#include <keccak/keccak.h>
#include <monolith/monolith.h>
#include <poseidon/poseidon.h>
#include <poseidon2/poseidon2.h>
#include <poseidon/poseidon_bn128.h>
#include <merkle/merkle.h>
#include <merkle/merkle_c.h>

/**
 * define DEBUG for printing
 */
// #define DEBUG

/**
 * define TIMING for printing execution time info
 */
// #define TIMING

#ifdef TIMING
#include <time.h>
#include <sys/time.h>
#endif  // TIMING

#ifdef DEBUG
void printhash(u64 *h)
{
    for (int i = 0; i < 4; i++)
    {
        printf("%lu ", h[i]);
    }
    printf("\n");
}
#endif

#ifdef USE_CUDA

#include <utils/cuda_utils.cuh>

__global__ void keccak_gpu_driver(u64 *input, u32 size, u64 *hash)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1)
        return;
    gpu_keccak_hash_one((gl64_t *)input, size, (gl64_t *)hash);
}

void keccak_hash_on_gpu(u64 *input, u32 size, u64 *hash)
{
    u64 *gpu_data, *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, size * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, input, size * sizeof(u64), cudaMemcpyHostToDevice));
    keccak_gpu_driver<<<1, 1>>>(gpu_data, size, gpu_hash);
    CHECKCUDAERR(cudaMemcpy(hash, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_data));
    CHECKCUDAERR(cudaFree(gpu_hash));
}

__global__ void monolith_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_monolith_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void monolith_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_monolith_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void monolith_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_monolith_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_poseidon_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_poseidon_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_poseidon_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon2_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_poseidon2_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon2_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_poseidon2_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon2_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    gpu_poseidon2_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidonbn128_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_poseidon_bn128_hash_one((gl64_t *)in, n, (gl64_t *)out);
}
#endif

TEST(LIBCUDA, keccak_test)
{
    u64 data[6] = {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 13588825262671526271ul, 13421290117754017454ul, 7401888676587830362ul};

    u64 expected[7][4] = {
        {0},
        {13421290117754017454ul, 0, 0, 0},
        {13421290117754017454ul, 7401888676587830362ul, 0, 0},
        {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 0},
        {9981707860959651334ul, 16351366398560378420ul, 4283762868800363615ul, 101},
        {708367124667950404ul, 17208681281141108820ul, 8334320481120086961ul, 134},
        {16109761546392287110ul, 4918745475135463511ul, 17110319063854316944ul, 103}};

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    for (int size = 1; size <= 6; size++)
    {
        cpu_keccak_hash_one(data, size, h1);
#ifdef USE_CUDA
        keccak_hash_on_gpu(data, size, h2);
#endif
#ifdef DEBUG
        printf("*** Size %d\n", size);
        printhash(h1);
        printhash(h2);
#endif
        for (int j = 0; j < 4; j++)
        {
            assert(h1[j] == expected[size][j]);
            assert(h2[j] == expected[size][j]);
        }
    }
}

TEST(LIBCUDA, monolith_test1)
{
    u64 inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    u64 exp[4] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    cpu_monolith_hash_one(inp, 12, h1);
#ifdef DEBUG
    printhash(h1);
#endif
    assert(h1[0] == exp[0]);
    assert(h1[1] == exp[1]);
    assert(h1[2] == exp[2]);
    assert(h1[3] == exp[3]);

#ifdef USE_CUDA
    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    monolith_hash<<<1, 1>>>(gpu_data, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(h2, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    printhash(h2);
#endif
    assert(h2[0] == exp[0]);
    assert(h2[1] == exp[1]);
    assert(h2[2] == exp[2]);
    assert(h2[3] == exp[3]);
#endif
}

#ifdef USE_CUDA
TEST(LIBCUDA, monolith_test2)
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

    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    monolith_hash_step1<<<1, 4>>>(gpu_data, gpu_hash, 7, 4);
    monolith_hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    monolith_hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < 28; i++)
    {
        assert(tree1[i] == tree2[i]);
    }
}
#endif // USE_CUDA

TEST(LIBCUDA, poseidon_test1)
{
    u64 leaf[7] = {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027};

    u64 expected[7][4] = {
        {0},
        {0},
        {8395359103262935841, 1377884553022145855, 0, 0},
        {0},
        {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162},
        {0},
        {16643938361881363776, 6653675298471110559, 12562058402463703932, 16154511938222758243},
    };

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    for (int k = 2; k <= 6; k += 2)
    {
        cpu_poseidon_hash_one(leaf, k, h1);
#ifdef DEBUG
        printhash(h1);
#endif

        for (int j = 0; j < 4; j++)
        {
            assert(h1[j] == expected[k][j]);
        }
    }

#ifdef USE_CUDA
    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 6 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, leaf, 6 * sizeof(u64), cudaMemcpyHostToDevice));

    for (int k = 2; k <= 6; k += 2)
    {
        poseidon_hash<<<1, 1>>>(gpu_leaf, gpu_hash, k);
        CHECKCUDAERR(cudaMemcpy(h2, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef DEBUG
        printhash(h2);
#endif // DEBUG

        for (int j = 0; j < 4; j++)
        {
            assert(h2[j] == expected[k][j]);
        }
    }
#endif // USE_CUDA

#ifdef RUST_POSEIDON
    ext_poseidon_hash_or_noop(h1, leaf, 6);
    printhash(h1);
#endif // RUST_POSEIDON
}

#ifdef USE_CUDA
TEST(LIBCUDA, poseidon_test2)
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
        cpu_poseidon_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    cpu_poseidon_hash_two(tree1, tree1 + 4, tree1 + 16);
    cpu_poseidon_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    cpu_poseidon_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

    // GPU
    u64 tree2[28] = {0ul};

    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon_hash_step1<<<1, 4>>>(gpu_leaf, gpu_hash, 7, 4);
    poseidon_hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    poseidon_hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < 28; i++)
    {
        assert(tree1[i] == tree2[i]);
    }
}
#endif // USE_CUDA

TEST(LIBCUDA, monolith_test3)
{
    u64 inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    u64 hash[4] = {0};
    u64 ref[4] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};

    cpu_monolith_hash_one(inp, 12, hash);
    for (int i = 0; i < 4; i++)
    {
        assert(hash[i] == ref[i]);
    }

#ifdef USE_CUDA
    u64 *gpu_inp;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_inp, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_inp, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    monolith_hash<<<1, 1>>>(gpu_inp, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(hash, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++)
    {
        assert(hash[i] == ref[i]);
    }
#endif
}

TEST(LIBCUDA, poseidon2_test1)
{
    // similar to the test in goldilocks repo
    // test 1 - Fibonacci
    u64 inp[12];
    inp[0] = 0;
    inp[1] = 1;
    for (int i = 2; i < 12; i++)
    {
        inp[i] = inp[i - 2] + inp[i - 1];
    }

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    cpu_poseidon2_hash_one(inp, 12, h1);
#ifdef DEBUG
    printhash(h1);
#endif

    assert(h1[0] == 0x133a03eca11d93fb);
    assert(h1[1] == 0x5365414fb618f58d);
    assert(h1[2] == 0xfa49f50f3a2ba2e5);
    assert(h1[3] == 0xd16e53672c9832a4);

#ifdef USE_CUDA
    u64 *gpu_inp;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_inp, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_inp, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_hash<<<1, 1>>>(gpu_inp, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(h2, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    printhash(h2);
#endif

    assert(h2[0] == 0x133a03eca11d93fb);
    assert(h2[1] == 0x5365414fb618f58d);
    assert(h2[2] == 0xfa49f50f3a2ba2e5);
    assert(h2[3] == 0xd16e53672c9832a4);
#endif
}

#ifdef USE_CUDA
TEST(LIBCUDA, poseidon2_test2)
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

    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_hash_step1<<<1, 4>>>(gpu_data, gpu_hash, 7, 4);
    poseidon2_hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    poseidon2_hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < 28; i++)
    {
        assert(tree1[i] == tree2[i]);
    }
}
#endif // USE_CUDA

TEST(LIBCUDA, poseidonbn128_test1)
{
    u64 inp[12] = {8917524657281059100u,
				   13029010200779371910u,
				   16138660518493481604u,
				   17277322750214136960u,
				   1441151880423231822u,
				   0, 0, 0, 0, 0, 0, 0};

    u64 exp[4] = {2163910501769503938, 9976732063159483418, 662985512748194034, 3626198389901409849};

    u64 cpu_out[HASH_SIZE_U64];

    cpu_poseidon_bn128_hash_one(inp, 12, cpu_out);

#ifdef DEBUG
    printhash(cpu_out);
#endif

    assert(cpu_out[0] == exp[0]);
    assert(cpu_out[1] == exp[1]);
    assert(cpu_out[2] == exp[2]);
    assert(cpu_out[3] == exp[3]);

#ifdef USE_CUDA
    u64 gpu_out[HASH_SIZE_U64];
    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidonbn128_hash<<<1, 1>>>(gpu_data, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(gpu_out, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    printhash(gpu_out);
#endif

    assert(gpu_out[0] == exp[0]);
    assert(gpu_out[1] == exp[1]);
    assert(gpu_out[2] == exp[2]);
    assert(gpu_out[3] == exp[3]);
#endif // USE_CUDA
}

#ifdef USE_CUDA
void compare_results(u64 *digests_buf1, u64 *digests_buf2, u32 n_digests, u64 *cap_buf1, u64 *cap_buf2, u32 n_caps)
{
    u64 *ptr1 = digests_buf1;
    u64 *ptr2 = digests_buf2;
    for (int i = 0; i < n_digests * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        assert(*ptr1 == *ptr2);
    }
    ptr1 = cap_buf1;
    ptr2 = cap_buf2;
    for (int i = 0; i < n_caps * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        assert(*ptr1 == *ptr2);
    }
}
/*
 * Run on GPU and CPU and compare the results. They have to be the same.
 */
#define LOG_SIZE    2
#define LEAF_SIZE_U64 6

TEST(LIBCUDA, merkle_test2)
// int x()
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 n_leaves = (1 << LOG_SIZE);
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    u64* digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64* cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64* leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));

    // Generate random leaves
    srand(time(NULL));
    for (int i = 0; i < n_leaves; i++)
    {
        for (int j = 0; j < LEAF_SIZE_U64; j++)
        {
            u32 r = rand();
            leaves_buf[i * LEAF_SIZE_U64 + j] = (u64)r << 32 + r * 88958514;
        }
    }
#ifdef DEBUG
    printf("Leaves count: %ld\n", n_leaves);
    printf("Leaf size: %d\n", LEAF_SIZE_U64);
    printf("Digests count: %ld\n", n_digests);
    printf("Caps count: %ld\n", n_caps);
    printf("Caps height: %ld\n", cap_h);
#endif // DEBUG

    // Compute on GPU
    u64 *gpu_leaves;
    u64 *gpu_digests;
    u32 *gpu_caps;

    u64* digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64* cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));

    CHECKCUDAERR(cudaMalloc(&gpu_leaves, n_leaves * LEAF_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_digests, n_digests * HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_caps, n_caps * HASH_SIZE_U64 * sizeof(u64)));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    CHECKCUDAERR(cudaMemcpy(gpu_leaves, leaves_buf, n_leaves * LEAF_SIZE_U64 * sizeof(u64), cudaMemcpyHostToDevice));
    fill_digests_buf_linear_gpu_with_gpu_ptr(
        gpu_digests,
        gpu_caps,
        gpu_leaves,
        n_digests,
        n_caps,
        n_leaves,
        LEAF_SIZE_U64,
        cap_h,
        HashType::HashPoseidon,
        0
    );
    CHECKCUDAERR(cudaMemcpy(digests_buf2, gpu_digests, n_digests * HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(cap_buf2, gpu_caps, n_caps * HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef TIMING
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on GPU: %ld us\n", elapsed);
#endif
    CHECKCUDAERR(cudaFree(gpu_leaves));
    CHECKCUDAERR(cudaFree(gpu_digests));
    CHECKCUDAERR(cudaFree(gpu_caps));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, LEAF_SIZE_U64, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

    compare_results(digests_buf1, digests_buf2, n_digests, cap_buf1, cap_buf2, n_caps);

    free(digests_buf1);
    free(digests_buf2);
    free(cap_buf1);
    free(cap_buf2);
}
#endif // USE_CUDA

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}