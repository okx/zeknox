#include "monolith.hpp"
#include "gl64_t.cuh"

#define LOOKUP_BITS 8
#define SPONGE_WIDTH 12
#define MONOLITH_N_ROUNDS 6

#ifdef USE_CUDA
__device__ __constant__ uint64_t GPU_MONOLITH_ROUND_CONSTANTS[MONOLITH_N_ROUNDS + 1][SPONGE_WIDTH] = {
#else
const uint64_t MONOLITH_ROUND_CONSTANTS[N_ROUNDS + 1][SPONGE_WIDTH] = {
#endif
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {
        13596126580325903823,
        5676126986831820406,
        11349149288412960427,
        3368797843020733411,
        16240671731749717664,
        9273190757374900239,
        14446552112110239438,
        4033077683985131644,
        4291229347329361293,
        13231607645683636062,
        1383651072186713277,
        8898815177417587567,
    },
    {
        2383619671172821638,
        6065528368924797662,
        16737578966352303081,
        2661700069680749654,
        7414030722730336790,
        18124970299993404776,
        9169923000283400738,
        15832813151034110977,
        16245117847613094506,
        11056181639108379773,
        10546400734398052938,
        8443860941261719174,
    },
    {
        15799082741422909885,
        13421235861052008152,
        15448208253823605561,
        2540286744040770964,
        2895626806801935918,
        8644593510196221619,
        17722491003064835823,
        5166255496419771636,
        1015740739405252346,
        4400043467547597488,
        5176473243271652644,
        4517904634837939508,
    },
    {
        18341030605319882173,
        13366339881666916534,
        6291492342503367536,
        10004214885638819819,
        4748655089269860551,
        1520762444865670308,
        8393589389936386108,
        11025183333304586284,
        5993305003203422738,
        458912836931247573,
        5947003897778655410,
        17184667486285295106,
    },
    {
        15710528677110011358,
        8929476121507374707,
        2351989866172789037,
        11264145846854799752,
        14924075362538455764,
        10107004551857451916,
        18325221206052792232,
        16751515052585522105,
        15305034267720085905,
        15639149412312342017,
        14624541102106656564,
        3542311898554959098,
    },
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

#ifdef USE_CUDA
__device__ __constant__ uint32_t GPU_MONOLITH_MAT_12[SPONGE_WIDTH][SPONGE_WIDTH] = {
#else
const uint32_t MONOLITH_MAT_12[SPONGE_WIDTH][SPONGE_WIDTH] = {
#endif
    {7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8},
    {8, 7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21},
    {21, 8, 7, 23, 8, 26, 13, 10, 9, 7, 6, 22},
    {22, 21, 8, 7, 23, 8, 26, 13, 10, 9, 7, 6},
    {6, 22, 21, 8, 7, 23, 8, 26, 13, 10, 9, 7},
    {7, 6, 22, 21, 8, 7, 23, 8, 26, 13, 10, 9},
    {9, 7, 6, 22, 21, 8, 7, 23, 8, 26, 13, 10},
    {10, 9, 7, 6, 22, 21, 8, 7, 23, 8, 26, 13},
    {13, 10, 9, 7, 6, 22, 21, 8, 7, 23, 8, 26},
    {26, 13, 10, 9, 7, 6, 22, 21, 8, 7, 23, 8},
    {8, 26, 13, 10, 9, 7, 6, 22, 21, 8, 7, 23},
    {23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8, 7},
};

#ifdef USE_CUDA
__device__ __forceinline__
#else
inline
#endif
    uint64_t
    bar_u64(uint64_t limb)
{
    uint64_t limbl1 = ((~limb & 0x8080808080808080) >> 7) | ((~limb & 0x7F7F7F7F7F7F7F7F) << 1); // Left rotation by 1
    uint64_t limbl2 = ((limb & 0xC0C0C0C0C0C0C0C0) >> 6) | ((limb & 0x3F3F3F3F3F3F3F3F) << 2);   // Left rotation by 2
    uint64_t limbl3 = ((limb & 0xE0E0E0E0E0E0E0E0) >> 5) | ((limb & 0x1F1F1F1F1F1F1F1F) << 3);   // Left rotation by 3

    // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
    uint64_t tmp = limb ^ (limbl1 & limbl2 & limbl3);
    return ((tmp & 0x8080808080808080) >> 7) | ((tmp & 0x7F7F7F7F7F7F7F7F) << 1);
}

#ifdef USE_CUDA
__device__ __forceinline__
#else
inline
#endif
    void
    bars(uint64_t *state)
{
    state[0] = bar_u64(state[0]);
    state[1] = bar_u64(state[1]);
    state[2] = bar_u64(state[2]);
    state[3] = bar_u64(state[3]);
}

// sl = state low (64 bits)
// sh = state high (output only and can be only 0 or 1)
#ifdef USE_CUDA
__device__ __forceinline__
#else
inline
#endif
    void
    bricks_u64(uint64_t *sl, uint32_t *sh)
{
    // Feistel Type-3
    // Use "& 0xFFFFFFFFFFFFFFFF" to tell the compiler it is dealing with 64-bit values (save
    // some instructions for upper half)
    for (int i = SPONGE_WIDTH - 1; i > 0; i--)
    {
#ifdef USE_CUDA
        gl64_t prev = (gl64_t)sl[i - 1];
        gl64_t tmp1 = prev * prev;
        uint64_t tmp2 = tmp1.get_val();
#else
        GoldilocksField prev = GoldilocksField(sl[i - 1]);
        GoldilocksField tmp1 = prev * prev;
        uint64_t tmp2 = tmp1.get_val();
#endif
        sl[i] += tmp2;
        sh[i] = (sl[i] < tmp2);
    }
}

#ifdef USE_CUDA
__device__ __forceinline__
#else
inline
#endif
    void
    concrete_u64(uint64_t *sl, uint32_t *sh, const uint64_t *rc)
{
    // temp storage
    uint64_t ssl[SPONGE_WIDTH];

    for (int row = 0; row < SPONGE_WIDTH; row++)
    {
        uint128_t res = 0;
        for (int column = 0; column < SPONGE_WIDTH; column++)
        {
#ifdef USE_CUDA
            res += ((uint128_t)sl[column]) * GPU_MONOLITH_MAT_12[row][column];
            uint32_t res_h = sh[column] * GPU_MONOLITH_MAT_12[row][column];
#else
            res += ((uint128_t)sl[column]) * MONOLITH_MAT_12[row][column];
            uint32_t res_h = sh[column] * MONOLITH_MAT_12[row][column];
#endif
            res += ((uint128_t)res_h) << 64;
        }
        res += rc[row];
#ifdef USE_CUDA
        uint32_t tmp[4] = {
            (uint32_t)res.lo, (uint32_t)(res.lo >> 32), (uint32_t)res.hi, (uint32_t)(res.hi >> 32)};
        gl64_t g = gl64_t(tmp);
        ssl[row] = g.get_val();
#else
        ssl[row] = GoldilocksField::from_noncanonical_u96((uint64_t)res, (uint32_t)(res >> 64)).get_val();
#endif
    }
    for (int row = 0; row < SPONGE_WIDTH; row++)
    {
        sl[row] = ssl[row];
        sh[row] = 0;
    }
}

#ifdef USE_CUDA
__device__ __forceinline__
#else
inline
#endif
    void
    monolith(uint64_t *state)
{
    uint32_t state_h[SPONGE_WIDTH] = {0};

#ifdef USE_CUDA
    concrete_u64(state, state_h, GPU_MONOLITH_ROUND_CONSTANTS[0]);
#else
    concrete_u64(state, state_h, MONOLITH_ROUND_CONSTANTS[0]);
#endif
    for (int r = 1; r < MONOLITH_N_ROUNDS + 1; r++)
    {
        bars(state);
        bricks_u64(state, state_h);
#ifdef USE_CUDA
        concrete_u64(state, state_h, GPU_MONOLITH_ROUND_CONSTANTS[r]);
#else
        concrete_u64(state, state_h, MONOLITH_ROUND_CONSTANTS[r]);
#endif
    }
}

#ifdef USE_CUDA
__device__ void MonolithPermutationGPU::permute2()
{
    monolith((uint64_t *)state);
}

__device__ void MonolithHasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
    if (num_inputs <= NUM_HASH_OUT_ELTS)
    {
        u32 i = 0;
        for (; i < num_inputs; i++)
        {
            hash[i] = inputs[i];
        }
        for (; i < NUM_HASH_OUT_ELTS; i++)
        {
            hash[i].zero();
        }
    }
    else
    {
        MonolithPermutationGPU perm = MonolithPermutationGPU();

        // Absorb all input chunks.
        for (u32 idx = 0; idx < num_inputs; idx += SPONGE_RATE)
        {
            perm.set_from_slice(inputs + idx, MIN(SPONGE_RATE, num_inputs - idx), 0);
            perm.permute2();
        }
        gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
        for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
        {
            hash[i] = ret[i];
        }
    }
}

__device__ void MonolithHasher::gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    MonolithPermutationGPU perm = MonolithPermutationGPU();
    perm.set_from_slice(hash1, NUM_HASH_OUT_ELTS, 0);
    perm.set_from_slice(hash2, NUM_HASH_OUT_ELTS, NUM_HASH_OUT_ELTS);
    perm.permute2();
    gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        hash[i] = ret[i];
    }
}

#else  // USE_CUDA

inline void MonolithPermutation::permute2()
{
    monolith((uint64_t *)state);
}

void MonolithHasher::cpu_hash_one(u64 *data, u64 data_size, u64 *digest)
{
    if (data_size <= NUM_HASH_OUT_ELTS)
    {
        for (u32 i = 0; i < data_size; i++)
        {
            digest[i] = data[i];
        }
        for (u32 i = data_size; i < NUM_HASH_OUT_ELTS; i++)
        {
            digest[i] = 0;
        }
        return;
    }
    GoldilocksField *in = (GoldilocksField *)malloc(data_size * sizeof(GoldilocksField));
    for (u32 i = 0; i < data_size; i++)
    {
        in[i] = GoldilocksField(data[i]);
    }
    MonolithPermutation perm = MonolithPermutation();
    u64 idx = 0;
    while (idx < data_size)
    {
        perm.set_from_slice(in + idx, MIN(SPONGE_RATE, (data_size - idx)), 0);
        perm.permute2();
        idx += SPONGE_RATE;
    }
    HashOut out = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        digest[i] = out.elements[i].get_val();
    }
    free(in);
}

void MonolithHasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
    HashOut x = HashOut(digest_left, NUM_HASH_OUT_ELTS);
    HashOut y = HashOut(digest_right, NUM_HASH_OUT_ELTS);
    MonolithPermutation perm = MonolithPermutation();
    perm.set_from_slice(x.elements, x.n_elements, 0);
    perm.set_from_slice(y.elements, y.n_elements, NUM_HASH_OUT_ELTS);
    perm.permute2();
    HashOut out = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        digest[i] = out.elements[i].get_val();
    }
}
#endif // USE_CUDA