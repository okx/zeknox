// keccak.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

#include "keccak.h"

#include <string.h>

#define KECCAK_ROUNDS 24
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

const uint64_t keccakf_rndc[24] = 
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

const int keccakf_rotc[24] = 
{
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

const int keccakf_piln[24] = 
{
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};

// update the state with given number of rounds

void keccakf(uint64_t st[25], int rounds)
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
            j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, keccakf_rotc[i]);
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
        st[0] ^= keccakf_rndc[round];
    }
}

// compute a keccak hash (md) of given byte length from "in"

int keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen)
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
        keccakf(st, KECCAK_ROUNDS);
    }
    
    // last block and padding
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++)
        st[i] ^= ((uint64_t *) temp)[i];

    keccakf(st, KECCAK_ROUNDS);

    memcpy(md, st, mdlen);

    return 0;
}

void cpu_keccak_hash_one(uint64_t* digest, uint64_t* data, uint32_t data_size) {
    if (data_size < 4) {
        memcpy(digest, data, data_size * 8);
        return;
    }
    keccak((uint8_t*)data, data_size * 8, (uint8_t*)digest, 32);
    digest[3] &= 0xFF;
}

void cpu_keccak_hash_two(uint64_t* digest, uint64_t* digest_left, uint64_t* digest_right) {
    uint64_t input[8];
    input[0] = digest_left[0];
    input[1] = digest_left[1];
    input[2] = digest_left[2];
    input[3] = digest_left[3];
    input[4] = digest_right[0];
    input[5] = digest_right[1];
    input[6] = digest_right[2];
    input[7] = digest_right[3];

    keccak((uint8_t*)input, 64, (uint8_t*)digest, 32);
    digest[3] &= 0xFF;
}