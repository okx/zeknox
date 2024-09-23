// Copyright 2024 OKX
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
    1431286215153372998ull, 3509349009260703107ull, 2289575380984896342ull, 10625215922958251110ull, 17137022507167291684ull, 17143426961497010024ull, 9589775313463224365ull, 7736066733515538648ull, 2217569167061322248ull, 10394930802584583083ull, 4612393375016695705ull, 5332470884919453534ull,
    8724526834049581439ull, 17673787971454860688ull, 2519987773101056005ull, 7999687124137420323ull, 18312454652563306701ull, 15136091233824155669ull, 1257110570403430003ull, 5665449074466664773ull, 16178737609685266571ull, 52855143527893348ull, 8084454992943870230ull, 2597062441266647183ull,
    3342624911463171251ull, 6781356195391537436ull, 4697929572322733707ull, 4179687232228901671ull, 17841073646522133059ull, 18340176721233187897ull, 13152929999122219197ull, 6306257051437840427ull, 4974451914008050921ull, 11258703678970285201ull, 581736081259960204ull, 18323286026903235604ull,
    10250026231324330997ull, 13321947507807660157ull, 13020725208899496943ull, 11416990495425192684ull, 7221795794796219413ull, 2607917872900632985ull, 2591896057192169329ull, 10485489452304998145ull, 9480186048908910015ull, 2645141845409940474ull, 16242299839765162610ull, 12203738590896308135ull,
    5395176197344543510ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    17941136338888340715ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    7559392505546762987ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    549633128904721280ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    15658455328409267684ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    10078371877170729592ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    2349868247408080783ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    13105911261634181239ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    12868653202234053626ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    9471330315555975806ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    4580289636625406680ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    13222733136951421572ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    4555032575628627551ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    7619130111929922899ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    4547848507246491777ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    5662043532568004632ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    15723873049665279492ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    13585630674756818185ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    6990417929677264473ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    6373257983538884779ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    1005856792729125863ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    17850970025369572891ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull,
    14306783492963476045ull, 12653264875831356889ull, 10887434669785806501ull, 7221072982690633460ull, 9953585853856674407ull, 13497620366078753434ull, 18140292631504202243ull, 17311934738088402529ull, 6686302214424395771ull, 11193071888943695519ull, 10233795775801758543ull, 3362219552562939863ull,
    8595401306696186761ull, 7753411262943026561ull, 12415218859476220947ull, 12517451587026875834ull, 3257008032900598499ull, 2187469039578904770ull, 657675168296710415ull, 8659969869470208989ull, 12526098871288378639ull, 12525853395769009329ull, 15388161689979551704ull, 7880966905416338909ull,
    2911694411222711481ull, 6420652251792580406ull, 323544930728360053ull, 11718666476052241225ull, 2449132068789045592ull, 17993014181992530560ull, 15161788952257357966ull, 3788504801066818367ull, 1282111773460545571ull, 8849495164481705550ull, 8380852402060721190ull, 2161980224591127360ull,
    2440151485689245146ull, 17521895002090134367ull, 13821005335130766955ull, 17513705631114265826ull, 17068447856797239529ull, 17964439003977043993ull, 5685000919538239429ull, 11615940660682589106ull, 2522854885180605258ull, 12584118968072796115ull, 17841258728624635591ull, 10821564568873127316ull};

#define ROUNDS_F 8
#define ROUNDS_P 22

#ifdef USE_CUDA
__device__ __forceinline__ void apply_m_4(gl64_t *x)
#else
inline void apply_m_4(GoldilocksField *x)
#endif
{
    auto t0 = x[0] + x[1];
    auto t1 = x[2] + x[3];
    auto t2 = x[1] + x[1] + t1;
    auto t3 = x[3] + x[3] + t0;
    auto t4 = t1 + t1 + t1 + t1 + t3;
    auto t5 = t0 + t0 + t0 + t0 + t2;
    auto t6 = t3 + t5;
    auto t7 = t2 + t4;
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
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
