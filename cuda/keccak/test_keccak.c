#include "types.h"

#define RUST_POSEIDON

#include "keccak.h"
#include "keccak-tiny.h"

#include <stdio.h>

void cpu_keccak_hash_one(u64* digest, u64* data, u32 data_size) {
    u8* ptr = (u8*)data;
    for (int i = 0; i < data_size * 8; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
    // sha3_256((u8*)digest, 25, (u8*)data, data_size * 8);
    shake256((u8*)digest, 25, (u8*)data, data_size * 8);
}

void cpu_keccak_hash_two(u64* digest, u64* digest_left, u64* digest_right) {
    
}

void printhash(u64* h) {
    for (int i = 0; i < 4; i++) {
        printf("%lu ", h[i]);
    }
    printf("\n");
}

int main() {

    u64 leaf[6] = {13421290117754017454, 7401888676587830362, 15316685236050041751, 13588825262671526271, 13421290117754017454, 7401888676587830362};

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    ext_keccak_hash_or_noop(h1, leaf, 1);
    cpu_keccak_hash_one(h2, leaf, 1);
    printhash(h1);
    printhash(h2);

    ext_keccak_hash_or_noop(h1, leaf, 4);
    cpu_keccak_hash_one(h2, leaf, 4);
    printhash(h1);
    printhash(h2);

    ext_keccak_hash_or_noop(h1, leaf, 6);
    cpu_keccak_hash_one(h2, leaf, 6);
    printhash(h1);
    printhash(h2);    

    return 0;
}