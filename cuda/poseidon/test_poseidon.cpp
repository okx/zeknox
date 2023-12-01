#include "poseidon.h"

#include "hash.h"

#include <stdio.h>

void printh(u64* h) {
    for (int i = 0; i < 4; i++) {
        printf("%lu ", h[i]);
    }
    printf("\n");
}
int main() {

    u64 leaf[6] = {13421290117754017454, 7401888676587830362, 15316685236050041751, 13588825262671526271, 13421290117754017454, 7401888676587830362};

    u64 h1[4] = {0u};
    u64 h2[4] = {0u};

    ext_hash_or_noop(h1, leaf, 1);
    compute_hash_leaf(h2, leaf, 1);
    printh(h1);
    printh(h2);

    ext_hash_or_noop(h1, leaf, 4);
    compute_hash_leaf(h2, leaf, 4);
    printh(h1);
    printh(h2);

    ext_hash_or_noop(h1, leaf, 6);
    compute_hash_leaf(h2, leaf, 6);
    printh(h1);
    printh(h2);    

    return 0;
}