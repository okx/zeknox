#include <gtest/gtest.h>
#include <assert.h>

#include <keccak/keccak.hpp>
#include <monolith/monolith.hpp>
#include <poseidon/poseidon.hpp>
#include <poseidon2/poseidon2.hpp>
#include <poseidon/poseidon_bn128.hpp>
#include <merkle/merkle.h>
#include <poseidon2/poseidon2_constants.hpp>

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
#endif // TIMING

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

/*
 * CUDA Kernels
 */
#ifdef USE_CUDA

#include <utils/cuda_utils.cuh>

__global__ void keccak_gpu_driver(u64 *input, u32 size, u64 *hash)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1)
        return;

    KeccakHasher::gpu_hash_one((gl64_t *)input, size, (gl64_t *)hash);
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

    MonolithHasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void monolith_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    MonolithHasher::gpu_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void monolith_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    MonolithHasher::gpu_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    PoseidonHasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    PoseidonHasher::gpu_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    PoseidonHasher::gpu_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon2_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    Poseidon2Hasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon2_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    Poseidon2Hasher::gpu_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon2_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    Poseidon2Hasher::gpu_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidonbn128_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    PoseidonBN128Hasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon2_goldilocks_width_8_zeroes(u64 *ptr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gl64_t state[8];

    hl_poseidon2_goldilocks_width_8(state);

    for (u64 i = 0; i < 8; i++)
    {
        ptr[i] = state[i].to_u64();
    }
}

__global__ void poseidon2_goldilocks_width_8_range(u64 *ptr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gl64_t state[8];
    for (u64 i = 0; i < 8; i++)
    {
        state[i] = gl64_t(i);
    }

    hl_poseidon2_goldilocks_width_8(state);

    for (u64 i = 0; i < 8; i++)
    {
        ptr[i] = state[i].to_u64();
    }
}

__global__ void poseidon2_goldilocks_width_8_random(u64 *ptr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gl64_t state[8];
    for (u64 i = 0; i < 8; i++)
    {
        state[i] = gl64_t(ptr[i]);
    }

    hl_poseidon2_goldilocks_width_8(state);

    for (u64 i = 0; i < 8; i++)
    {
        ptr[i] = state[i].to_u64();
    }
}

template <const u64 WIDTH>
__global__ void poseidon2_babybear_random(u64 *ptr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    u32 *external_consts_ptr = (WIDTH == 16) ? GPU_BABYBEAR_WIDTH_16_EXT_CONST_P3 : GPU_BABYBEAR_WIDTH_24_EXT_CONST_P3;
    u32 *internal_consts_ptr = (WIDTH == 16) ? GPU_BABYBEAR_WIDTH_16_INT_CONST_P3 : GPU_BABYBEAR_WIDTH_24_INT_CONST_P3;

    const u64 internal_consts_size = (WIDTH == 16) ? 13 : 21;

    BabyBearField external_consts[8 * WIDTH];
    for (u64 i = 0; i < 8 * WIDTH; i++)
    {
        external_consts[i] = BabyBearField(external_consts_ptr[i]);
    }
    BabyBearField internal_consts[internal_consts_size];
    for (u64 i = 0; i < internal_consts_size; i++)
    {
        internal_consts[i] = BabyBearField(internal_consts_ptr[i]);
    }
    BabyBearField state[WIDTH];
    for (u64 i = 0; i < WIDTH; i++)
    {
        state[i] = BabyBearField((u32)ptr[i]);
    }

    auto external_matrix = Poseidon2ExternalMatrixGeneral<BabyBearField, WIDTH>();
    auto diffusion_matrix = BabyBearDiffusionMatrix<WIDTH>();
    auto poseidon2 = Poseidon2<BabyBearField, Poseidon2ExternalMatrixGeneral<BabyBearField, WIDTH>, BabyBearDiffusionMatrix<WIDTH>, WIDTH, 7>((u64)8, (u64)internal_consts_size, external_consts, internal_consts, external_matrix, diffusion_matrix);

    poseidon2.permute_mut(state);

    for (u64 i = 0; i < WIDTH; i++)
    {
        ptr[i] = state[i].to_u64();
    }
}

#endif // USE_CUDA

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
#ifdef USE_CUDA
    u64 h2[4] = {0u};
#endif

    for (int size = 1; size <= 6; size++)
    {
        KeccakHasher::cpu_hash_one(data, size, h1);
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
#ifdef USE_CUDA
            assert(h2[j] == expected[size][j]);
#endif
        }
    }
}

TEST(LIBCUDA, monolith_test1)
{
    u64 inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    u64 exp[4] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};

    u64 h1[4] = {0u};

    MonolithHasher::cpu_hash_one(inp, 12, h1);
#ifdef DEBUG
    printhash(h1);
#endif
    assert(h1[0] == exp[0]);
    assert(h1[1] == exp[1]);
    assert(h1[2] == exp[2]);
    assert(h1[3] == exp[3]);

#ifdef USE_CUDA
    u64 h2[4] = {0u};
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
        MonolithHasher::cpu_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    MonolithHasher::cpu_hash_two(tree1, tree1 + 4, tree1 + 16);
    MonolithHasher::cpu_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    MonolithHasher::cpu_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

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
    u64 leaf[9] = {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027, 16154511938222758243, 3651132590097252162};

    u64 expected[11][4] = {
        {0},
        {8395359103262935841, 0, 0, 0},
        {8395359103262935841, 1377884553022145855, 0, 0},
        {8395359103262935841, 1377884553022145855, 2370707998790318766, 0},
        {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162},
        {3618821072812614426, 8353148445756493727, 4040525329700581442, 15983474240847269257},
        {16643938361881363776, 6653675298471110559, 12562058402463703932, 16154511938222758243},
        {7544909477878586743, 7431000548126831493, 17815668806142634286, 13168106265494210017},
        {6835933650993053111, 15978194778874965616, 2024081381896137659, 16520693669262110264},
        {9429914239539731992, 14881719063945231827, 15528667124986963891, 16465743531992249573},
        {16643938361881363776, 6653675298471110559, 12562058402463703932, 16154511938222758243}};

    u64 h1[4] = {0u};

    for (int k = 1; k <= 9; k++)
    {
        PoseidonHasher::cpu_hash_one(leaf, k, h1);
#ifdef DEBUG
        printhash(h1);
#endif
        for (int j = 0; j < 4; j++)
        {
            assert(h1[j] == expected[k][j]);
        }
    }

#ifdef USE_CUDA
    u64 h2[4] = {0u};
    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 9 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, leaf, 9 * sizeof(u64), cudaMemcpyHostToDevice));

    for (int k = 1; k <= 9; k++)
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
        PoseidonHasher::cpu_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    PoseidonHasher::cpu_hash_two(tree1, tree1 + 4, tree1 + 16);
    PoseidonHasher::cpu_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    PoseidonHasher::cpu_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

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

    MonolithHasher::cpu_hash_one(inp, 12, hash);
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

    Poseidon2Hasher hasher;
    hasher.cpu_hash_one(inp, 12, h1);
#ifdef DEBUG
    printhash(h1);
#endif

    assert(h1[0] == 0x133a03eca11d93fb);
    assert(h1[1] == 0x5365414fb618f58d);
    assert(h1[2] == 0xfa49f50f3a2ba2e5);
    assert(h1[3] == 0xd16e53672c9832a4);

#ifdef USE_CUDA
    u64 h2[4] = {0u};
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

TEST(LIBCUDA, cuda_poseidon2_goldilocks_width_8_zeroes)
{
    u64 out[8] = {0};

    u64 expected[8] = {
        4214787979728720400,
        12324939279576102560,
        10353596058419792404,
        15456793487362310586,
        10065219879212154722,
        16227496357546636742,
        2959271128466640042,
        14285409611125725709,
    };

    u64 *gpu_out;
    CHECKCUDAERR(cudaMalloc(&gpu_out, 8 * sizeof(u64)));
    poseidon2_goldilocks_width_8_zeroes<<<1, 1>>>(gpu_out);
    CHECKCUDAERR(cudaMemcpy(out, gpu_out, 8 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u64 i = 0; i < 8; i++)
    {
        assert(out[i] == expected[i]);
    }
}

TEST(LIBCUDA, cuda_poseidon2_goldilocks_width_8_range)
{
    u64 out[8] = {0};

    u64 expected[8] = {
        14266028122062624699,
        5353147180106052723,
        15203350112844181434,
        17630919042639565165,
        16601551015858213987,
        10184091939013874068,
        16774100645754596496,
        12047415603622314780,
    };

    u64 *gpu_out;
    CHECKCUDAERR(cudaMalloc(&gpu_out, 8 * sizeof(u64)));
    poseidon2_goldilocks_width_8_range<<<1, 1>>>(gpu_out);
    CHECKCUDAERR(cudaMemcpy(out, gpu_out, 8 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u64 i = 0; i < 8; i++)
    {
        assert(out[i] == expected[i]);
    }
}

TEST(LIBCUDA, cuda_poseidon2_goldilocks_width_8_random)
{
    u64 input[8] = {
        5116996373749832116,
        8931548647907683339,
        17132360229780760684,
        11280040044015983889,
        11957737519043010992,
        15695650327991256125,
        17604752143022812942,
        543194415197607509};

    u64 expected[8] = {
        1831346684315917658,
        13497752062035433374,
        12149460647271516589,
        15656333994315312197,
        4671534937670455565,
        3140092508031220630,
        4251208148861706881,
        6973971209430822232,
    };

    u64 *gpu_out;
    CHECKCUDAERR(cudaMalloc(&gpu_out, 8 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_out, input, 8 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_goldilocks_width_8_random<<<1, 1>>>(gpu_out);
    CHECKCUDAERR(cudaMemcpy(input, gpu_out, 8 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u64 i = 0; i < 8; i++)
    {
        assert(input[i] == expected[i]);
    }
}

TEST(LIBCUDA, cuda_poseidon2_babybear_width_16_random)
{
    constexpr u64 _width = 16;

    u64 input[_width] = {
        894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
        120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
        1783754913};

    u64 expected[_width] = {
        512585766, 975869435, 1921378527, 1238606951, 899635794, 132650430, 1426417547,
        1734425242, 57415409, 67173027, 1535042492, 1318033394, 1070659233, 17258943,
        856719028, 1500534995};

    u64 *gpu_out;
    CHECKCUDAERR(cudaMalloc(&gpu_out, _width * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_out, input, _width * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_babybear_random<_width><<<1, 1>>>(gpu_out);
    CHECKCUDAERR(cudaMemcpy(input, gpu_out, _width * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_out));

    for (u64 i = 0; i < _width; i++)
    {
        assert(input[i] == expected[i]);
    }
}

TEST(LIBCUDA, cuda_poseidon2_babybear_width_24_random)
{
    constexpr u64 _width = 24;

    u64 input[_width] = {
        886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
        355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
        1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 449439011,
        1131357108, 50869465, 1589724894};

    u64 expected[_width] = {
        162275163, 462059149, 1096991565, 924509284, 300323988, 608502870, 427093935,
        733126108, 1676785000, 669115065, 441326760, 60861458, 124006210, 687842154, 270552480,
        1279931581, 1030167257, 126690434, 1291783486, 669126431, 1320670824, 1121967237,
        458234203, 142219603};

    u64 *gpu_out;
    CHECKCUDAERR(cudaMalloc(&gpu_out, _width * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_out, input, _width * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_babybear_random<_width><<<1, 1>>>(gpu_out);
    CHECKCUDAERR(cudaMemcpy(input, gpu_out, _width * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_out));

    for (u64 i = 0; i < _width; i++)
    {
        assert(input[i] == expected[i]);
    }
}

#else // USE_CUDA

TEST(LIBCUDA, poseidon2_goldilocks_test_width_8_zeroes)
{
    GoldilocksField input[8] = {GoldilocksField::Zero()};

    u64 expected[8] = {
        4214787979728720400,
        12324939279576102560,
        10353596058419792404,
        15456793487362310586,
        10065219879212154722,
        16227496357546636742,
        2959271128466640042,
        14285409611125725709,
    };

    hl_poseidon2_goldilocks_width_8(input);

    for (u64 i = 0; i < 8; i++)
    {
        assert(input[i].to_noncanonical_u64() == expected[i]);
    }
}

TEST(LIBCUDA, poseidon2_goldilocks_test_width_8_range)
{
    GoldilocksField input[8];

    for (u64 i = 0; i < 8; i++)
    {
        input[i] = GoldilocksField::from_canonical_u64(i);
    }

    u64 expected[8] = {
        14266028122062624699,
        5353147180106052723,
        15203350112844181434,
        17630919042639565165,
        16601551015858213987,
        10184091939013874068,
        16774100645754596496,
        12047415603622314780,
    };

    hl_poseidon2_goldilocks_width_8(input);

    for (u64 i = 0; i < 8; i++)
    {
        assert(input[i].to_noncanonical_u64() == expected[i]);
    }
}

TEST(LIBCUDA, poseidon2_goldilocks_test_width_8_random)
{
    GoldilocksField input[8] = {
        5116996373749832116,
        8931548647907683339,
        17132360229780760684,
        11280040044015983889,
        11957737519043010992,
        15695650327991256125,
        17604752143022812942,
        543194415197607509};

    u64 expected[8] = {
        1831346684315917658,
        13497752062035433374,
        12149460647271516589,
        15656333994315312197,
        4671534937670455565,
        3140092508031220630,
        4251208148861706881,
        6973971209430822232,
    };

    hl_poseidon2_goldilocks_width_8(input);

    for (u64 i = 0; i < 8; i++)
    {
        assert(input[i].to_noncanonical_u64() == expected[i]);
    }
}

TEST(LIBCUDA, poseidon2_babybear_width_16_random)
{
    BabyBearField input[16] = {
        894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
        120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
        1783754913};

    u64 expected[16] = {
        512585766, 975869435, 1921378527, 1238606951, 899635794, 132650430, 1426417547,
        1734425242, 57415409, 67173027, 1535042492, 1318033394, 1070659233, 17258943,
        856719028, 1500534995};

    BabyBearField external_consts[128];
    for (u64 i = 0; i < 128; i++)
    {
        external_consts[i] = BabyBearField(BABYBEAR_WIDTH_16_EXT_CONST_P3[i]);
    }
    BabyBearField internal_consts[13];
    for (u64 i = 0; i < 13; i++)
    {
        internal_consts[i] = BabyBearField(BABYBEAR_WIDTH_16_INT_CONST_P3[i]);
    }

    auto external_matrix = Poseidon2ExternalMatrixGeneral<BabyBearField, 16>();
    auto diffusion_matrix = BabyBearDiffusionMatrix<16>();
    auto poseidon2 = Poseidon2<BabyBearField, Poseidon2ExternalMatrixGeneral<BabyBearField, 16>, BabyBearDiffusionMatrix<16>, 16, 7>((u64)8, (u64)13, external_consts, internal_consts, external_matrix, diffusion_matrix);

    poseidon2.permute_mut(input);

    for (u64 i = 0; i < 16; i++)
    {
        assert(input[i].to_u64() == expected[i]);
    }
}

TEST(LIBCUDA, poseidon2_babybear_width_24_random)
{
    BabyBearField input[24] = {
        886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
        355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
        1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 449439011,
        1131357108, 50869465, 1589724894};

    u64 expected[24] = {
        162275163, 462059149, 1096991565, 924509284, 300323988, 608502870, 427093935,
        733126108, 1676785000, 669115065, 441326760, 60861458, 124006210, 687842154, 270552480,
        1279931581, 1030167257, 126690434, 1291783486, 669126431, 1320670824, 1121967237,
        458234203, 142219603};

    BabyBearField external_consts[192];
    for (u64 i = 0; i < 192; i++)
    {
        external_consts[i] = BabyBearField(BABYBEAR_WIDTH_24_EXT_CONST_P3[i]);
    }
    BabyBearField internal_consts[21];
    for (u64 i = 0; i < 21; i++)
    {
        internal_consts[i] = BabyBearField(BABYBEAR_WIDTH_24_INT_CONST_P3[i]);
    }

    auto external_matrix = Poseidon2ExternalMatrixGeneral<BabyBearField, 24>();
    auto diffusion_matrix = BabyBearDiffusionMatrix<24>();
    auto poseidon2 = Poseidon2<BabyBearField, Poseidon2ExternalMatrixGeneral<BabyBearField, 24>, BabyBearDiffusionMatrix<24>, 24, 7>((u64)8, (u64)21, external_consts, internal_consts, external_matrix, diffusion_matrix);

    poseidon2.permute_mut(input);

    for (u64 i = 0; i < 24; i++)
    {
        assert(input[i].to_u64() == expected[i]);
    }
}

#endif // ifndef USE_CUDA

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

    Poseidon2Hasher hasher;
    for (u32 i = 0; i < 4; i++)
    {
        hasher.cpu_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    hasher.cpu_hash_two(tree1, tree1 + 4, tree1 + 16);
    hasher.cpu_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    hasher.cpu_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

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

    PoseidonBN128Hasher hasher;
    hasher.cpu_hash_one(inp, 12, cpu_out);

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
#ifdef DEBUG
    for (int i = 0; i < n_digests; i++, ptr1 += HASH_SIZE_U64, ptr2 += HASH_SIZE_U64)
    {
        printf("Hashes digests\n");
        printhash(ptr1);
        printhash(ptr2);
    }
    ptr1 = digests_buf1;
    ptr2 = digests_buf2;
#endif
    for (int i = 0; i < n_digests * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        assert(*ptr1 == *ptr2);
    }
    ptr1 = cap_buf1;
    ptr2 = cap_buf2;
#ifdef DEBUG
    for (int i = 0; i < n_caps; i++, ptr1 += HASH_SIZE_U64, ptr2 += HASH_SIZE_U64)
    {
        printf("Hashes digests\n");
        printhash(ptr1);
        printhash(ptr2);
    }
    ptr1 = cap_buf1;
    ptr2 = cap_buf2;
#endif
    for (int i = 0; i < n_caps * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        assert(*ptr1 == *ptr2);
    }
}
/*
 * Run on GPU and CPU and compare the results. They have to be the same.
 */
#define LOG_SIZE 2
#define LEAF_SIZE_U64 6

TEST(LIBCUDA, merkle_test2)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 n_leaves = (1 << LOG_SIZE);
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));

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

    u64 *digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));

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
        0);
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

TEST(LIBCUDA, merkle_test3)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 leaf_size = 7;
    u64 n_leaves = 4;
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    // u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *digests_buf1 = cap_buf1;
    u64 *leaves_buf = (u64 *)malloc(n_leaves * leaf_size * sizeof(u64));

    u64 leaf[7] = {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027};
    memcpy(leaves_buf, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 7, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 14, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 21, leaf, 7 * sizeof(u64));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, leaf_size, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

#ifdef DEBUG
    printf("Digests:\n");
    for (int i = 0; i < n_digests; i++)
    {
        printhash(digests_buf1 + i * HASH_SIZE_U64);
    }
    printf("Caps:\n");
    for (int i = 0; i < n_caps; i++)
    {
        printhash(cap_buf1 + i * HASH_SIZE_U64);
    }
#endif
    free(cap_buf1);
    free(leaves_buf);
}

#ifdef __USE_AVX__
TEST(LIBCUDA, merkle_avx_test3)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 leaf_size = 7;
    u64 n_leaves = 4;
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    // u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *digests_buf1 = cap_buf1;
    u64 *leaves_buf = (u64 *)malloc(n_leaves * leaf_size * sizeof(u64));

    u64 leaf[7] = {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027};
    memcpy(leaves_buf, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 7, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 14, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 21, leaf, 7 * sizeof(u64));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu_avx(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, leaf_size, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

#ifdef DEBUG
    printf("Digests:\n");
    for (int i = 0; i < n_digests; i++)
    {
        printhash(digests_buf1 + i * HASH_SIZE_U64);
    }
    printf("Caps:\n");
    for (int i = 0; i < n_caps; i++)
    {
        printhash(cap_buf1 + i * HASH_SIZE_U64);
    }
#endif
    free(cap_buf1);
    free(leaves_buf);
}
#endif // __USE_AVX__

#ifdef __AVX512__
TEST(LIBCUDA, merkle_avx512_test3)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 leaf_size = 7;
    u64 n_leaves = 4;
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    // u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *digests_buf1 = cap_buf1;
    u64 *leaves_buf = (u64 *)malloc(n_leaves * leaf_size * sizeof(u64));

    u64 leaf[7] = {8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027};
    memcpy(leaves_buf, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 7, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 14, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 21, leaf, 7 * sizeof(u64));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu_avx512(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, leaf_size, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

#ifdef DEBUG
    printf("Digests:\n");
    for (int i = 0; i < n_digests; i++)
    {
        printhash(digests_buf1 + i * HASH_SIZE_U64);
    }
    printf("Caps:\n");
    for (int i = 0; i < n_caps; i++)
    {
        printhash(cap_buf1 + i * HASH_SIZE_U64);
    }
#endif
    free(cap_buf1);
    free(leaves_buf);
}
#endif // __AVX512__

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}