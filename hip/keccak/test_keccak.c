#include "keccak_cpu.h"

#include <stdio.h>

void printhash(uint64_t *h)
{
    for (int i = 0; i < 4; i++)
    {
        printf("%lu ", h[i]);
    }
    printf("\n");
}

int main()
{

    uint64_t leaf[6] = {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 13588825262671526271ul, 13421290117754017454ul, 7401888676587830362ul};

    uint64_t h1[4] = {0u};

    for (int size = 1; size <= 6; size++)
    {
        printf("*** Size %d\n", size);
        keccak((uint8_t *)leaf, size, (uint8_t *)h1, 32);
        printhash(h1);
    }

    return 0;
}