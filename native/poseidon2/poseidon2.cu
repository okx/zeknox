// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "poseidon2.hpp"

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_MATRIX_DIAG_12_GOLDILOCKS[12] = {
#else
const u64 MATRIX_DIAG_12_GOLDILOCKS[12] = {
#endif
    0xc3b6c08e23ba9300,
    0xd84b5de94a324fb6,
    0x0d0c371c5b35b84f,
    0x7964f570e7188037,
    0x5daf18bbd996604b,
    0x6743bc47b9595257,
    0x5528b9362c59bb70,
    0xac45e25b7127b68b,
    0xa2077d7dfbb606b5,
    0xf3faac6faee378ae,
    0x0c6388b51545e883,
    0xd27dbb6944917b60};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_RC12[360] = {
#else
const u64 RC12[360] = {
#endif
    15492826721047263190ull, 11728330187201910315ull, 8836021247773420868ull, 16777404051263952451ull, 5510875212538051896ull, 6173089941271892285ull, 2927757366422211339ull, 10340958981325008808ull, 8541987352684552425ull, 9739599543776434497ull, 15073950188101532019ull, 12084856431752384512ull,
    4584713381960671270ull, 8807052963476652830ull, 54136601502601741ull, 4872702333905478703ull, 5551030319979516287ull, 12889366755535460989ull, 16329242193178844328ull, 412018088475211848ull, 10505784623379650541ull, 9758812378619434837ull, 7421979329386275117ull, 375240370024755551ull,
    3331431125640721931ull, 15684937309956309981ull, 578521833432107983ull, 14379242000670861838ull, 17922409828154900976ull, 8153494278429192257ull, 15904673920630731971ull, 11217863998460634216ull, 3301540195510742136ull, 9937973023749922003ull, 3059102938155026419ull, 1895288289490976132ull,
    5580912693628927540ull, 10064804080494788323ull, 9582481583369602410ull, 10186259561546797986ull, 247426333829703916ull, 13193193905461376067ull, 6386232593701758044ull, 17954717245501896472ull, 1531720443376282699ull, 2455761864255501970ull, 11234429217864304495ull, 4746959618548874102ull,
    11921381764981422944ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    10318423381711320787ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    8291411502347000766ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    229948027109387563ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    9152521390190983261ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    7129306032690285515ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    15395989607365232011ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    8641397269074305925ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    17256848792241043600ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    6046475228902245682ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    12041608676381094092ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    12785542378683951657ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    14546032085337914034ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    3304199118235116851ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    16499627707072547655ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    10386478025625759321ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    13475579315436919170ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    16042710511297532028ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    1411266850385657080ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    9024840976168649958ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    14047056970978379368ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    838728605080212101ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    13571697342473846203ull, 17477857865056504753ull, 15963032953523553760ull, 16033593225279635898ull, 14252634232868282405ull, 8219748254835277737ull, 7459165569491914711ull, 15855939513193752003ull, 16788866461340278896ull, 7102224659693946577ull, 3024718005636976471ull, 13695468978618890430ull,
    8214202050877825436ull, 2670727992739346204ull, 16259532062589659211ull, 11869922396257088411ull, 3179482916972760137ull, 13525476046633427808ull, 3217337278042947412ull, 14494689598654046340ull, 15837379330312175383ull, 8029037639801151344ull, 2153456285263517937ull, 8301106462311849241ull,
    13294194396455217955ull, 17394768489610594315ull, 12847609130464867455ull, 14015739446356528640ull, 5879251655839607853ull, 9747000124977436185ull, 8950393546890284269ull, 10765765936405694368ull, 14695323910334139959ull, 16366254691123000864ull, 15292774414889043182ull, 10910394433429313384ull,
    17253424460214596184ull, 3442854447664030446ull, 3005570425335613727ull, 10859158614900201063ull, 9763230642109343539ull, 6647722546511515039ull, 909012944955815706ull, 18101204076790399111ull, 11588128829349125809ull, 15863878496612806566ull, 5201119062417750399ull, 176665553780565743ull};

#define ROUNDS_F 8
#define ROUNDS_P 22

#ifdef USE_CUDA
__device__ __forceinline__ void apply_m_4(gl64_t *x)
#else
inline void apply_m_4(GoldilocksField *x)
#endif
{
    auto t01 = x[0] + x[1];
    auto t23 = x[2] + x[3];
    auto t0123 = t01 + t23;
    auto t01123 = t0123 + x[1];
    auto t01233 = t0123 + x[3];
    auto new_x3 = t01233 + x[0] + x[0];
    auto new_x1 = t01123 + x[2] + x[2];
    auto new_x0 = t01123 + t01;
    auto new_x2 = t01233 + t23;
    x[0] = new_x0;
    x[1] = new_x1;
    x[2] = new_x2;
    x[3] = new_x3;
}

#ifdef USE_CUDA
__device__ __forceinline__ void add_rc(gl64_t *state, gl64_t *rc)
#else
inline void add_rc(GoldilocksField *state, GoldilocksField *rc)
#endif
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + rc[i];
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ gl64_t sbox_p(gl64_t &x)
#else
inline GoldilocksField sbox_p(GoldilocksField &x)
#endif
{
    auto x2 = x * x;
    auto x4 = x2 * x2;
    auto x3 = x2 * x;
    return x3 * x4;
}

#ifdef USE_CUDA
__device__ __forceinline__ void sbox(gl64_t *state)
#else
inline void sbox(GoldilocksField *state)
#endif
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = sbox_p(state[i]);
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void ext_permute_mut(gl64_t *state)
#else
inline void ext_permute_mut(GoldilocksField *state)
#endif
{
    for (u32 i = 0; i < SPONGE_WIDTH; i += 4)
    {
        apply_m_4(state + i);
    }

#ifdef USE_CUDA
    gl64_t sums[4];
#else
    GoldilocksField sums[4];
#endif
    sums[0] = state[0] + state[4] + state[8];
    sums[1] = state[1] + state[5] + state[9];
    sums[2] = state[2] + state[6] + state[10];
    sums[3] = state[3] + state[7] + state[11];

    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + sums[i % 4];
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void matmul_internal(gl64_t *state, gl64_t *mat_internal_diag_m_1)
#else
inline void matmul_internal(GoldilocksField *state, GoldilocksField *mat_internal_diag_m_1)
#endif
{
    auto sum = state[0];
    for (u32 i = 1; i < SPONGE_WIDTH; i++)
    {
        sum = sum + state[i];
    }

    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] * mat_internal_diag_m_1[i];
        state[i] = state[i] + sum;
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void poseidon2(gl64_t *state)
#else
inline void poseidon2(GoldilocksField *state)
#endif
{
#ifdef USE_CUDA
    gl64_t *rc12 = (gl64_t *)GPU_RC12;
    gl64_t *md12 = (gl64_t *)GPU_MATRIX_DIAG_12_GOLDILOCKS;
#else
    GoldilocksField *rc12 = (GoldilocksField *)RC12;
    GoldilocksField *md12 = (GoldilocksField *)MATRIX_DIAG_12_GOLDILOCKS;
#endif

    // The initial linear layer.
    ext_permute_mut(state);

    // The first half of the external rounds.
    u32 rounds = ROUNDS_F + ROUNDS_P;
    u32 rounds_f_beginning = ROUNDS_F / 2;
    for (u32 r = 0; r < rounds_f_beginning; r++)
    {
        add_rc(state, &rc12[12 * r]);
        sbox(state);
        ext_permute_mut(state);
    }

    // The internal rounds.
    u32 p_end = rounds_f_beginning + ROUNDS_P;
    for (u32 r = rounds_f_beginning; r < p_end; r++)
    {
        state[0] = state[0] + rc12[12 * r];
        state[0] = sbox_p(state[0]);
        matmul_internal(state, md12);
    }

    // The second half of the external rounds.
    for (u32 r = p_end; r < rounds; r++)
    {
        add_rc(state, &rc12[12 * r]);
        sbox(state);
        ext_permute_mut(state);
    }
}

#ifdef USE_CUDA
__forceinline__ __device__ void Poseidon2PermutationGPU::permute()
{
    poseidon2(get_state());
}

__device__ void Poseidon2Hasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
    PoseidonPermutationGPU::gpu_hash_one_with_permutation_template<Poseidon2PermutationGPU>(inputs, num_inputs, hash);
}

__device__ void Poseidon2Hasher::gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    PoseidonPermutationGPU::gpu_hash_two_with_permutation_template<Poseidon2PermutationGPU>(hash1, hash2, hash);
}

#else // USE_CUDA

inline void Poseidon2Permutation::permute()
{
    poseidon2(get_state());
}

void Poseidon2Hasher::cpu_hash_one(u64 *input, u64 input_count, u64 *digest)
{
    PoseidonPermutation::cpu_hash_one_with_permutation_template<Poseidon2Permutation>(input, input_count, digest);
}

void Poseidon2Hasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
    PoseidonPermutation::cpu_hash_two_with_permutation_template<Poseidon2Permutation>(digest_left, digest_right, digest);
}

#endif // USE_CUDA
