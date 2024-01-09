#ifndef __CRYPTO_ARITHMATIC_HPP__
#define __CRYPTO_ARITHMATIC_HPP__

#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif

#if defined(FEATURE_GOLDILOCKS)
#include <arithmetic/gl64.cu>
#elif defined(FEATURE_BN254)
#include <arithmetic/bn254.cu>
#endif
#endif