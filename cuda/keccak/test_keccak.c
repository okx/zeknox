#include "int_types.h"

#define RUST_POSEIDON

#include "keccak.h"

#include <stdio.h>

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

    for (int size = 1; size <= 6; size++) {
        printf("*** Size %d\n", size);
        ext_keccak_hash_or_noop(h1, leaf, size);
        cpu_keccak_hash_one(h2, leaf, size);
        printhash(h1);
        printhash(h2);
    }  

    return 0;
}