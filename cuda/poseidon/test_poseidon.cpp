#include "int_types.h"

#include "poseidon.h"

#include <stdio.h>

void printhash(u64* h) {
    for (int i = 0; i < 4; i++) {
        printf("%lu ", h[i]);
    }
    printf("\n");
}

int test1() {
    u64 leaf[6] = {13421290117754017454, 7401888676587830362, 15316685236050041751, 13588825262671526271, 13421290117754017454, 7401888676587830362};

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

/*
#ifdef RUST_POSEIDON
    ext_poseidon_hash_or_noop(h1, leaf, 1);
    printhash(h1);
#endif
    cpu_poseidon_hash_one(leaf, 1, h2);
    printhash(h2);

#ifdef RUST_POSEIDON
    ext_poseidon_hash_or_noop(h1, leaf, 4);
    printhash(h1);
#endif
    cpu_poseidon_hash_one(leaf, 4, h2);
    printhash(h2);
*/

#ifdef RUST_POSEIDON
    ext_poseidon_hash_or_noop(h1, leaf, 6);
    printhash(h1);
#endif
    cpu_poseidon_hash_one(leaf, 6, h2);
    printhash(h2);

    return 1;
}

int main() {

    test1();

    return 0;
}