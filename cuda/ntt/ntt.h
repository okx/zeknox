#ifndef __CRYPTO_NTT_NTT_H__
#define __CRYPTO_NTT_NTT_H__

namespace Ntt_Types
{
    enum InputOutputOrder
    {
        NN,
        NR,
        RN,
        RR
    };
    enum Direction
    {
        forward,
        inverse
    };
    enum Type
    {
        standard,
        coset
    };
    enum Algorithm
    {
        GS,
        CT
    };
}

#endif