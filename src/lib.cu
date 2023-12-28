#include <cuda.h>
#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif

#if defined(FEATURE_GOLDILOCKS)

#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN128)
#include <ff/alt_bn128.hpp>
#else
#error "no FEATURE"
#endif

#include <util/cuda_available.hpp>
#include <ntt/ntt.cuh>
#include <ntt/ntt.h>
#include <arithmetic/arithmetic.hpp>
#ifndef __CUDA_ARCH__ // below is cpu code; __CUDA_ARCH__ should not be defined

#if defined(EXPOSE_C_INTERFACE)
extern "C"
#endif
    RustError
    compute_ntt(size_t device_id, fr_t *inout, uint32_t lg_domain_size,
                Ntt_Types::InputOutputOrder ntt_order,
                Ntt_Types::Direction ntt_direction,
                Ntt_Types::Type ntt_type)
{
    auto &gpu = select_gpu(device_id);

    return NTT::Base(gpu, inout, lg_domain_size,
                     ntt_order, ntt_direction, ntt_type);
}
#endif
