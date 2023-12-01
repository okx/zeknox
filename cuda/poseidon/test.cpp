#include "field.hpp"
#include <stdio.h>

int main() {
    // u64 a = 8139794479952933663;
    // u64 b = 11417638945989878436;
    // u64 b = 8139794479952933663;

    // u64 a = 1406232391689923845;
    // u64 b = 13797305789616547983;

    u64 a = 15075242217845939242;
    u64 b = 7167831807589368522;
    
    u64 c = GoldilocksField::mulmod(a, b);
    printf("Correct:\n%lu\n", c);

    // u64 c = GoldilocksField::mul(a, b);
    // printf("%lu\n", c);

    c = GoldilocksField::reduce128v3((u128)a * (u128)b);
    printf("%lu\n", c);
    
    c = GoldilocksField::reduce128((u128)a * (u128)b);
    printf("%lu\n", c);
   
    return 0;
}