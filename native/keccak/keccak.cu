// Adapted from keccak.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

#include "keccak/keccak.hpp"
#include "types/int_types.h"
#include "utils/cuda_utils.cuh"

#define KECCAK_ROUNDS 24
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

CONST u64 VAR(keccakf_rndc)[24] =
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

DEVICE void FUNC(keccakf)(u64 st[25], int rounds)
{
    int i, j, round;
    u64 t, bc[5];

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

DEVICE void FUNC(keccak)(const u8 *in, int inlen, u8 *md, int mdlen)
{
    u64 st[25];
    u8 temp[144];
    int i, rsiz, rsizw;

    rsiz = 200 - 2 * mdlen;
    rsizw = rsiz / 8;

    memset(st, 0, sizeof(st));

    for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (i = 0; i < rsizw; i++)
            st[i] ^= ((u64 *) in)[i];
        FUNC(keccakf)(st, KECCAK_ROUNDS);
    }

    // last block and padding
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++)
        st[i] ^= ((u64 *) temp)[i];

    FUNC(keccakf)(st, KECCAK_ROUNDS);

    memcpy(md, st, mdlen);
}

#ifdef USE_CUDA
#ifdef __CUDA_ARCH__
__device__ void KeccakHasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
    if (num_inputs < 4) {
        memcpy(hash, inputs, num_inputs * 8);
        return;
    }
    FUNC(keccak)((u8*)inputs, num_inputs * 8, (u8*)hash, 32);
    hash[3] &= 0xFF;
}

__device__ void KeccakHasher::gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    u8 input[50];
    u8* ileft = (u8*)hash1;
    u8* iright = (u8*)hash2;
    memcpy(input, ileft, 25);
    memcpy(input + 25, iright, 25);
    FUNC(keccak)((u8*)input, 50, (u8*)hash, 32);
    hash[3] &= 0xFF;
}
#endif
#else

void KeccakHasher::cpu_hash_one(u64* data, u64 data_size, u64* digest) {
    if (data_size < 4) {
        memcpy(digest, data, data_size * 8);
        return;
    }
    FUNC(keccak)((u8*)data, data_size * 8, (u8*)digest, 32);
    digest[3] &= 0xFF;
}

void KeccakHasher::cpu_hash_two(u64* digest_left, u64* digest_right, u64* digest) {
    u8 input[50];
    u8* ileft = (u8*)digest_left;
    u8* iright = (u8*)digest_right;
    memcpy(input, ileft, 25);
    memcpy(input + 25, iright, 25);
    FUNC(keccak)((u8*)input, 50, (u8*)digest, 32);
    digest[3] &= 0xFF;
}

#endif