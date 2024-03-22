#include "poseidon2.cuh"

#ifdef USE_CUDA
__device__ u64 GPU_MATRIX_DIAG_12_GOLDILOCKS[12] = {
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
    0xd27dbb6944917b60
};

#ifdef USE_CUDA
__device__ u64 GPU_RC12[360] = {
#else
const u64 RC12[360] = {
#endif
1431286215153372998, 3509349009260703107, 2289575380984896342, 10625215922958251110, 17137022507167291684, 17143426961497010024, 9589775313463224365, 7736066733515538648, 2217569167061322248, 10394930802584583083, 4612393375016695705, 5332470884919453534,
8724526834049581439, 17673787971454860688, 2519987773101056005, 7999687124137420323, 18312454652563306701, 15136091233824155669, 1257110570403430003, 5665449074466664773, 16178737609685266571, 52855143527893348, 8084454992943870230, 2597062441266647183,
3342624911463171251, 6781356195391537436, 4697929572322733707, 4179687232228901671, 17841073646522133059, 18340176721233187897, 13152929999122219197, 6306257051437840427, 4974451914008050921, 11258703678970285201, 581736081259960204, 18323286026903235604,
10250026231324330997, 13321947507807660157, 13020725208899496943, 11416990495425192684, 7221795794796219413, 2607917872900632985, 2591896057192169329, 10485489452304998145, 9480186048908910015, 2645141845409940474, 16242299839765162610, 12203738590896308135,
5395176197344543510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
17941136338888340715, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
7559392505546762987, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
549633128904721280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
15658455328409267684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
10078371877170729592, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
2349868247408080783, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
13105911261634181239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
12868653202234053626, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
9471330315555975806, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
4580289636625406680, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
13222733136951421572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
4555032575628627551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
7619130111929922899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
4547848507246491777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
5662043532568004632, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
15723873049665279492, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
13585630674756818185, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
6990417929677264473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
6373257983538884779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1005856792729125863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
17850970025369572891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
14306783492963476045, 12653264875831356889, 10887434669785806501, 7221072982690633460, 9953585853856674407, 13497620366078753434, 18140292631504202243, 17311934738088402529, 6686302214424395771, 11193071888943695519, 10233795775801758543, 3362219552562939863,
8595401306696186761, 7753411262943026561, 12415218859476220947, 12517451587026875834, 3257008032900598499, 2187469039578904770, 657675168296710415, 8659969869470208989, 12526098871288378639, 12525853395769009329, 15388161689979551704, 7880966905416338909,
2911694411222711481, 6420652251792580406, 323544930728360053, 11718666476052241225, 2449132068789045592, 17993014181992530560, 15161788952257357966, 3788504801066818367, 1282111773460545571, 8849495164481705550, 8380852402060721190, 2161980224591127360,
2440151485689245146, 17521895002090134367, 13821005335130766955, 17513705631114265826, 17068447856797239529, 17964439003977043993, 5685000919538239429, 11615940660682589106, 2522854885180605258, 12584118968072796115, 17841258728624635591, 10821564568873127316
};

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
__device__ __forceinline__ void add_rc(gl64_t *state, gl64_t* rc)
#else
inline void add_rc(GoldilocksField *state, GoldilocksField* rc)
#endif
{
    for(u32 i = 0; i < SPONGE_WIDTH; i++) {
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
    for (u32 i = 0; i < SPONGE_WIDTH; i++) {
        state[i] = sbox_p(state[i]);
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void ext_permute_mut(gl64_t *state)
#else
inline void ext_permute_mut(GoldilocksField *state)
#endif
{
        // First, we apply M_4 to each consecutive four elements of the state.
        // In Appendix B's terminology, this replaces each x_i with x_i'.
        for (u32 i = 0; i < SPONGE_WIDTH; i+=4) {
            apply_m_4(state + i);
        }

        // Now, we apply the outer circulant matrix (to compute the y_i values).

        // We first precompute the four sums of every four elements.
#ifdef USE_CUDA
    gl64_t sums[4];
#else
    GoldilocksField sums[4];
#endif
    sums[0] = state[0] + state[4] + state[8];
    sums[1] = state[1] + state[5] + state[9];
    sums[2] = state[2] + state[6] + state[10];
    sums[3] = state[3] + state[7] + state[11];

        // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
        // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
        for (u32 i = 0; i < SPONGE_WIDTH; i++) {
            state[i] = state[i] + sums[i % 4];
        }
    }

#ifdef USE_CUDA
__device__ __forceinline__ void matmul_internal(gl64_t *state, gl64_t* mat_internal_diag_m_1)
#else
inline void matmul_internal(GoldilocksField *state, GoldilocksField* mat_internal_diag_m_1)
#endif
 {
    auto sum = state[0];
    for (u32 i = 1; i < SPONGE_WIDTH; i++) {
        sum = sum + state[i];
    }

    for (u32 i = 0; i < SPONGE_WIDTH; i++) {
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
    gl64_t* rc12 = (gl64_t*)GPU_RC12;
    gl64_t* md12 = (gl64_t*)GPU_MATRIX_DIAG_12_GOLDILOCKS;
#else
    GoldilocksField* rc12 = (GoldilocksField*)RC12;
    GoldilocksField* md12 = (GoldilocksField*)MATRIX_DIAG_12_GOLDILOCKS;
#endif

        // The initial linear layer.
        ext_permute_mut(state);

        // The first half of the external rounds.
        u32 rounds = ROUNDS_F + ROUNDS_P;
        u32 rounds_f_beginning = ROUNDS_F / 2;
        for (u32 r = 0; r < rounds_f_beginning; r++) {
            add_rc(state, &rc12[12 * r]);
            sbox(state);
            ext_permute_mut(state);
        }

        // The internal rounds.
        u32 p_end = rounds_f_beginning + ROUNDS_P;
        for (u32 r = rounds_f_beginning; r < p_end; r++) {
            state[0] = state[0] + rc12[12 * r];
            state[0] = sbox_p(state[0]);
            matmul_internal(state, md12);
        }

        // The second half of the external rounds.
        for (u32 r = p_end; r < rounds; r++) {
            add_rc(state, &rc12[12 * r]);
            sbox(state);
            ext_permute_mut(state);
        }
    }

#ifdef USE_CUDA
__device__ void Poseidon2PermutationGPU::permute2()
{
    poseidon2(state);
}

__device__ void gpu_poseidon2_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
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
        Poseidon2PermutationGPU perm = Poseidon2PermutationGPU();

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

__device__ void gpu_poseidon2_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    Poseidon2PermutationGPU perm = Poseidon2PermutationGPU();
    perm.set_from_slice(hash1, NUM_HASH_OUT_ELTS, 0);
    perm.set_from_slice(hash2, NUM_HASH_OUT_ELTS, NUM_HASH_OUT_ELTS);
    perm.permute2();
    gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        hash[i] = ret[i];
    }
}

/*
void init_gpu_const() {
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_MATRIX_DIAG_12_GOLDILOCKS, MATRIX_DIAG_12_GOLDILOCKS, 12 * sizeof(u64), 0));
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_RC12, RC12, 360 * sizeof(u64), 0));
}
*/

#else
inline void Poseidon2Permutation::permute2()
{
    poseidon2(state);
}

void cpu_poseidon2_hash_one(u64 *data, u32 data_size, u64 *digest)
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
    Poseidon2Permutation perm = Poseidon2Permutation();
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

void cpu_poseidon2_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
    HashOut x = HashOut(digest_left, NUM_HASH_OUT_ELTS);
    HashOut y = HashOut(digest_right, NUM_HASH_OUT_ELTS);
    Poseidon2Permutation perm = Poseidon2Permutation();
    perm.set_from_slice(x.elements, x.n_elements, 0);
    perm.set_from_slice(y.elements, y.n_elements, NUM_HASH_OUT_ELTS);
    perm.permute2();
    HashOut out = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        digest[i] = out.elements[i].get_val();
    }
}
#endif