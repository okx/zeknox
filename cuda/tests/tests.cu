#include <gtest/gtest.h>
#include <assert.h>

#include <keccak/keccak.hpp>
#include <monolith/monolith.hpp>
#include <poseidon/poseidon.hpp>
#include <poseidon2/poseidon2.hpp>
#include <poseidon/poseidon_bn128.hpp>
#include <merkle/merkle.h>

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

#ifndef USE_CUDA

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

    BabyBearField external_consts[8 * 16] = {
        0x4EC2680C,
        0x110279CB,
        0x332D1F04,
        0x7DA39A8,
        0x20D60D25,
        0x6837F03,
        0x5C499950,
        0x11D53B04,
        0x4769DFB8,
        0x4D9FB1BF,
        0x21DD1576,
        0x58811E55,
        0x4D9AA329,
        0x2B412236,
        0x3822164A,
        0x1345CE59,
        0x3E8B66A4,
        0x3104F884,
        0x53CF147F,
        0x4C3C5D00,
        0xCCE8ACB,
        0x35EB8AB9,
        0x4EB90117,
        0xB0AB74,
        0x3DC5CED0,
        0x57E20602,
        0x5543D924,
        0x6A06354,
        0x4305BF7B,
        0x516D38B,
        0x288628B2,
        0xE801D9D,
        0x4FC7D3F6,
        0x5AC756E3,
        0x6DBC7A1D,
        0x22768C25,
        0x2D798FD,
        0x6AB7F8C8,
        0x610A3FA1,
        0x664B611A,
        0x61807B01,
        0x4277C420,
        0x429C112E,
        0x2873398E,
        0x74FD36A3,
        0x5180E3B0,
        0xB25B05,
        0x744737A6,
        0x1F828B47,
        0x73760EFB,
        0x571450E7,
        0x471CDEB5,
        0x48D335C3,
        0x46913BAD,
        0x19D6C553,
        0x72F92FBD,
        0x1E25480B,
        0x110700EF,
        0x3E6C0276,
        0x363288DE,
        0x4B0A4DD6,
        0x338B375D,
        0x124CD27E,
        0x1E178795,
        0x5777F011,
        0x68948CEB,
        0x19C2A7BC,
        0x5691B910,
        0x3491DC1C,
        0x20C91A20,
        0x56442FD6,
        0x37FE675F,
        0x6A4822D6,
        0x458DA37C,
        0x6688B5A,
        0x2299FBFF,
        0x776837E8,
        0x32EE44AA,
        0x37B964B7,
        0x3B4B31B4,
        0x6DBD1269,
        0x1A1F74C6,
        0x5C6B1DB1,
        0x3670308A,
        0x23D18114,
        0x22BFE022,
        0x4B432285,
        0x64E58ED,
        0x108AF480,
        0xA298030,
        0x15CE7F03,
        0x5831987C,
        0x56E222BA,
        0x61818A80,
        0x1DC0806D,
        0x2C2E3A27,
        0x26D36BF6,
        0x1BB4B661,
        0x558A76B,
        0x1575D881,
        0xDBA9003,
        0x69AFCEC8,
        0x52436DEB,
        0x2C6805C3,
        0x2F4B7A6E,
        0x512367CC,
        0x560DC002,
        0x3139B8A,
        0x2B987ECA,
        0x40D2C58A,
        0x4BEED74D,
        0x12925C15,
        0x29E264AA,
        0x57CFF845,
        0x7DFD045,
        0x505CD248,
        0x1C6B0406,
        0x55C4A053,
        0x252590,
        0x506BA70E,
        0x747EDFDA,
        0x690819DF,
        0x4BC54E23,
        0x5EF7512C,
        0x33870D43,
        0x84B39D1,
        0x3EC935D6,
        0x19340FFB,
    };

    BabyBearField internal_consts[13] = {0x1801CCD8, 0x131D9E83, 0x42EC25EE, 0x5FC787D, 0xC1356DB, 0x491AAE7C, 0x40E3021A, 0x3D25F0A, 0x68BDAFC, 0x2B32678F, 0x631CE19F, 0x2F8CC233, 0x401CE61F};

    auto external_matrix = Poseidon2ExternalMatrixGeneral<BabyBearField, 16>();
    auto diffusion_matrix = BabyBearDiffusionMatrix16();
    auto poseidon2 = Poseidon2<BabyBearField, Poseidon2ExternalMatrixGeneral<BabyBearField, 16>, BabyBearDiffusionMatrix16, 16, 7>((u64)8, (u64)13, external_consts, internal_consts, external_matrix, diffusion_matrix);

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

    BabyBearField external_consts[8 * 24] = {
        0x4EC2680C, 0x110279CB, 0x332D1F04, 0x7DA39A8, 0x20D60D25, 0x6837F03, 0x5C499950, 0x11D53B04, 0x4769DFB8, 0x4D9FB1BF, 0x21DD1576, 0x58811E55, 0x4D9AA329, 0x2B412236, 0x3822164A, 0x1345CE59, 0x3E8B66A4, 0x3104F884, 0x53CF147F, 0x4C3C5D00, 0xCCE8ACB, 0x35EB8AB9, 0x4EB90117, 0xB0AB74, 0x3DC5CED0, 0x57E20602, 0x5543D924, 0x6A06354, 0x4305BF7B, 0x516D38B, 0x288628B2, 0xE801D9D, 0x4FC7D3F6, 0x5AC756E3, 0x6DBC7A1D, 0x22768C25, 0x2D798FD, 0x6AB7F8C8, 0x610A3FA1, 0x664B611A, 0x61807B01, 0x4277C420, 0x429C112E, 0x2873398E, 0x74FD36A3, 0x5180E3B0, 0xB25B05, 0x744737A6, 0x1F828B47, 0x73760EFB, 0x571450E7, 0x471CDEB5, 0x48D335C3, 0x46913BAD, 0x19D6C553, 0x72F92FBD, 0x1E25480B, 0x110700EF, 0x3E6C0276, 0x363288DE, 0x4B0A4DD6, 0x338B375D, 0x124CD27E, 0x1E178795, 0x5777F011, 0x68948CEB, 0x19C2A7BC, 0x5691B910, 0x3491DC1C, 0x20C91A20, 0x56442FD6, 0x37FE675F, 0x6A4822D6, 0x458DA37C, 0x6688B5A, 0x2299FBFF, 0x776837E8, 0x32EE44AA, 0x37B964B7, 0x3B4B31B4, 0x6DBD1269, 0x1A1F74C6, 0x5C6B1DB1, 0x3670308A, 0x23D18114, 0x22BFE022, 0x4B432285, 0x64E58ED, 0x108AF480, 0xA298030, 0x15CE7F03, 0x5831987C, 0x56E222BA, 0x61818A80, 0x1DC0806D, 0x2C2E3A27, 0x26D36BF6, 0x1BB4B661, 0x558A76B, 0x1575D881, 0xDBA9003, 0x69AFCEC8, 0x52436DEB, 0x2C6805C3, 0x2F4B7A6E, 0x512367CC, 0x560DC002, 0x3139B8A, 0x2B987ECA, 0x40D2C58A, 0x4BEED74D, 0x12925C15, 0x29E264AA, 0x57CFF845, 0x7DFD045, 0x505CD248, 0x1C6B0406, 0x55C4A053, 0x252590, 0x506BA70E, 0x747EDFDA, 0x690819DF, 0x4BC54E23, 0x5EF7512C, 0x33870D43, 0x84B39D1, 0x3EC935D6, 0x19340FFB, 0x1801CCD8, 0x131D9E83, 0x42EC25EE, 0x5FC787D, 0xC1356DB, 0x491AAE7C, 0x40E3021A, 0x3D25F0A, 0x68BDAFC, 0x2B32678F, 0x631CE19F, 0x2F8CC233, 0x401CE61F, 0x5DE64A96, 0x15CD53A1, 0x24021AD3, 0x7466B2AB, 0x5DFBA9DE, 0x3F3B5642, 0x4DD9BBC1, 0x9CDE2AA, 0x73827615, 0x677A602B, 0x69BFF5DF, 0x6BCA4452, 0x538EACBB, 0x21695E2F, 0x48FD28A5, 0x53DB5C4C, 0x39AB7ABE, 0x60226CA8, 0x6F39CE27, 0x6DD72702, 0x61C72CA5, 0xB2ABE90, 0x3673352A, 0x36298C76, 0x50DE59D, 0x4169C3EE, 0x63258D2A, 0x59C45549, 0x3EB0408A, 0x72CE8221, 0x7372C616, 0x346F1D76, 0x42B0E84C, 0x271CB214, 0xF64F596, 0x2DEC45DF, 0x27FC1A0, 0x3C938ABF, 0x61BAD871, 0x6E5FD31D, 0x6F36A6D4, 0x544B3F0E, 0x18F27FA1, 0x34451992, 0x2417883F, 0x5157A5B6, 0x2EEB111E, 0x150135D7, 0x355925A3, 0x33329A06, 0x460CB30C};

    BabyBearField internal_consts[21] = {
        0x24D2AF3B,
        0x766ABFC3,
        0x4A4BBF41,
        0x18DEDD7,
        0x37A4705,
        0x27666A5D,
        0x3475E251,
        0xF0CB909,
        0x68ACF372,
        0x22CEC228,
        0x164774DC,
        0x59034D05,
        0x752865FD,
        0x64E74AD,
        0x5233240A,
        0x39C32A85,
        0x89480D7,
        0x3C439665,
        0x70908112,
        0x2C664E9E,
        0x647B04AC};

    auto external_matrix = Poseidon2ExternalMatrixGeneral<BabyBearField, 24>();
    auto diffusion_matrix = BabyBearDiffusionMatrix24();
    auto poseidon2 = Poseidon2<BabyBearField, Poseidon2ExternalMatrixGeneral<BabyBearField, 24>, BabyBearDiffusionMatrix24, 24, 7>((u64)8, (u64)21, external_consts, internal_consts, external_matrix, diffusion_matrix);

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