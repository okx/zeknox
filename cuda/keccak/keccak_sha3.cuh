// sha3.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>

// Revised 07-Aug-15 to match with official release of FIPS PUB 202 "SHA3"
// Revised 03-Sep-15 for portability + OpenSSL - style API

#include <stdio.h>
#include <stdint.h>
#include "cuda_utils.cuh"

#define TPB 128
#define MAX_RECORD  2048    // max 2048 bytes

#ifndef KECCAKF_ROUNDS
#define KECCAKF_ROUNDS 24
#endif

#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif

// state context
typedef struct
{
    union
    {                   // state:
        uint8_t b[200]; // 8-bit bytes
        uint64_t q[25]; // 64-bit words
    } st;
    int pt, rsiz, mdlen; // these don't overflow
} sha3_ctx_t;

// constants
const uint64_t host_keccakf_rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008};
const int host_keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
const int host_keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

__device__ __constant__ uint64_t gpu_keccakf_rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008};

__device__ __constant__ int gpu_keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};

__device__ __constant__ int gpu_keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

// update the state with given number of rounds

__host__ __device__ void sha3_keccakf(uint64_t st[25])
{
    // variables
    int i, j, r;
    uint64_t t, bc[5];

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    uint8_t *v;

    // endianess conversion. this is redundant on little-endian targets
    for (i = 0; i < 25; i++)
    {
        v = (uint8_t *)&st[i];
        st[i] = ((uint64_t)v[0]) | (((uint64_t)v[1]) << 8) |
                (((uint64_t)v[2]) << 16) | (((uint64_t)v[3]) << 24) |
                (((uint64_t)v[4]) << 32) | (((uint64_t)v[5]) << 40) |
                (((uint64_t)v[6]) << 48) | (((uint64_t)v[7]) << 56);
    }
#endif

    // actual iteration
    for (r = 0; r < KECCAKF_ROUNDS; r++)
    {

        // Theta
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++)
        {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // Rho Pi
        t = st[1];
        for (i = 0; i < 24; i++)
        {
        #ifdef  __CUDA_ARCH__
            j = gpu_keccakf_piln[i];
            int r = gpu_keccakf_rotc[i];
        #else
            j = hash_keccakf_piln[i];
            int r = host_keccakf_rotc[i];
        #endif
            bc[0] = st[j];
            st[j] = ROTL64(t, r);
            t = bc[0];
        }

        //  Chi
        for (j = 0; j < 25; j += 5)
        {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        //  Iota
        #ifdef  __CUDA_ARCH__
        st[0] ^= gpu_keccakf_rndc[r];
        #else
        st[0] ^= host_keccakf_rndc[r];
        #endif
    }

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    // endianess conversion. this is redundant on little-endian targets
    for (i = 0; i < 25; i++)
    {
        v = (uint8_t *)&st[i];
        t = st[i];
        v[0] = t & 0xFF;
        v[1] = (t >> 8) & 0xFF;
        v[2] = (t >> 16) & 0xFF;
        v[3] = (t >> 24) & 0xFF;
        v[4] = (t >> 32) & 0xFF;
        v[5] = (t >> 40) & 0xFF;
        v[6] = (t >> 48) & 0xFF;
        v[7] = (t >> 56) & 0xFF;
    }
#endif
}

// Initialize the context for SHA3

__host__ __device__ void sha3_init(sha3_ctx_t *c, int mdlen)
{
    int i;

    for (i = 0; i < 25; i++)
        c->st.q[i] = 0;
    c->mdlen = mdlen;
    c->rsiz = 200 - 2 * mdlen;
    c->pt = 0;
}

// update state with more data

__host__ __device__ void sha3_update(sha3_ctx_t *c, const void *data, size_t len)
{
    size_t i;
    int j;

    j = c->pt;
    for (i = 0; i < len; i++)
    {
        c->st.b[j++] ^= ((const uint8_t *)data)[i];
        if (j >= c->rsiz)
        {
            sha3_keccakf(c->st.q);
            j = 0;
        }
    }
    c->pt = j;
}

// finalize and output a hash
__host__ __device__ void sha3_final(void *md, sha3_ctx_t *c)
{
    int i;

    c->st.b[c->pt] ^= 0x06;
    c->st.b[c->rsiz - 1] ^= 0x80;
    sha3_keccakf(c->st.q);

    for (i = 0; i < c->mdlen; i++)
    {
        ((uint8_t *)md)[i] = c->st.b[i];
    }
}

__device__ void gpu_keccak_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{    
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // sha3_ctx_t *context = sha3_gpu_contexts + tid;
    sha3_ctx_t local_context;
    sha3_ctx_t *context = &local_context;
    
    sha3_init(context, 32);
    sha3_update(context, (void*)inputs, num_inputs * 8);
    sha3_final((void*)hash, context);
    hash[3] &= 0xFF;
}

__device__ void gpu_keccak_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // sha3_ctx_t *context = sha3_gpu_contexts + tid;
    sha3_ctx_t local_context;
    sha3_ctx_t *context = &local_context;

    gl64_t input[8];
    input[0] = hash1[0];
    input[1] = hash1[1];
    input[2] = hash1[2];
    input[3] = hash1[3];
    input[4] = hash2[0];
    input[5] = hash2[1];
    input[6] = hash2[2];
    input[7] = hash2[3];

    sha3_init(context, 32);
    sha3_update(context, (void*)input, 64);
    sha3_final((void*)hash, context);
    hash[3] &= 0xFF;
}

__host__ void cpu_keccak_hash_one(u64* digest, u64* data, u32 data_size) {
    sha3_ctx_t context;
    sha3_init(&context, 32);
    sha3_update(&context, (void*)data, data_size * 8);
    sha3_final((void*)digest, &context);
    digest[3] &= 0xFF;
}

__host__ void cpu_keccak_hash_two(u64* digest, u64* digest_left, u64* digest_right) {
    sha3_ctx_t context;

    u64 input[8];
    input[0] = digest_left[0];
    input[1] = digest_left[1];
    input[2] = digest_left[2];
    input[3] = digest_left[3];
    input[4] = digest_right[0];
    input[5] = digest_right[1];
    input[6] = digest_right[2];
    input[7] = digest_right[3];

    sha3_init(&context, 32);
    sha3_update(&context, (void*)input, 64);
    sha3_final((void*)digest, &context);
    digest[3] &= 0xFF;
}