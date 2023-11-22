#ifndef __CRYPTO_NTT_NTT_CUH__
#define __CRYPTO_NTT_NTT_CUH__

#include <util/gpu_t.cuh>
#ifndef __CUDA_ARCH__
class NTT {

public:
    enum class InputOutputOrder {NN, NR, RN, RR};
    enum class Direction {forward, inverse};
    enum class Type {standard, coset};
    enum class Algorithm {GS, CT};

protected:
    static void bit_rev(fr_t* d_out, const fr_t* d_inp, uint32_t lg_domain_size, stream_t& stream) {

    }

};

#endif
#endif