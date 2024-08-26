// Adapted from keccak.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

#include "keccak.hpp"
#include "int_types.h"
#include "cuda_utils.cuh"

#define KECCAK_ROUNDS 24
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

CONST uint64_t VAR(keccakf_rndc)[24] =
{
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

CONST int VAR(keccakf_rotc)[24] =
{
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

CONST int VAR(keccakf_piln)[24] =
{
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

// update the state with given number of rounds

DEVICE void FUNC(keccakf)(uint64_t st[25], int rounds)
{
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < rounds; round++) {

        // Theta
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // Rho Pi
        t = st[1];
        for (i = 0; i < 24; i++) {
            j = VAR(keccakf_piln)[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, VAR(keccakf_rotc)[i]);
            t = bc[0];
        }

        //  Chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        //  Iota
        st[0] ^= VAR(keccakf_rndc)[round];
    }
}

// compute a keccak hash (md) of given byte length from "in"

DEVICE void FUNC(keccak)(const uint8_t *in, int inlen, uint8_t *md, int mdlen)
{
    uint64_t st[25];
    uint8_t temp[144];
    int i, rsiz, rsizw;

    rsiz = 200 - 2 * mdlen;
    rsizw = rsiz / 8;

    memset(st, 0, sizeof(st));

    for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (i = 0; i < rsizw; i++)
            st[i] ^= ((uint64_t *) in)[i];
        FUNC(keccakf)(st, KECCAK_ROUNDS);
    }

    // last block and padding
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++)
        st[i] ^= ((uint64_t *) temp)[i];

    FUNC(keccakf)(st, KECCAK_ROUNDS);

    memcpy(md, st, mdlen);
}

#ifdef USE_CUDA
__device__ void KeccakHasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
    // assume num_inputs >= 4
    FUNC(keccak)((uint8_t*)inputs, num_inputs * 8, (uint8_t*)hash, 32);
    hash[3] &= 0xFF;
}

__device__ void KeccakHasher::gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    uint8_t input[50];
    uint8_t* ileft = (uint8_t*)hash1;
    uint8_t* iright = (uint8_t*)hash2;
    memcpy(input, ileft, 25);
    memcpy(input + 25, iright, 25);
    FUNC(keccak)((uint8_t*)input, 50, (uint8_t*)hash, 32);
    hash[3] &= 0xFF;
}

#else

void KeccakHasher::cpu_hash_one(uint64_t* data, uint64_t data_size, uint64_t* digest) {
    if (data_size < 4) {
        memcpy(digest, data, data_size * 8);
        return;
    }
    FUNC(keccak)((uint8_t*)data, data_size * 8, (uint8_t*)digest, 32);
    digest[3] &= 0xFF;
}

void KeccakHasher::cpu_hash_two(uint64_t* digest_left, uint64_t* digest_right, uint64_t* digest) {
    uint8_t input[50];
    uint8_t* ileft = (uint8_t*)digest_left;
    uint8_t* iright = (uint8_t*)digest_right;
    memcpy(input, ileft, 25);
    memcpy(input + 25, iright, 25);
    FUNC(keccak)((uint8_t*)input, 50, (uint8_t*)digest, 32);
    digest[3] &= 0xFF;
}

#endif