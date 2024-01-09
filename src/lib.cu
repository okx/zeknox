#include <cuda.h>
#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif

#if defined(FEATURE_GOLDILOCKS)

#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn254.hpp>
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

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_inf_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

// #define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>
#include <cstdio>
#include <blst_t.hpp>
// #if defined(EXPOSE_C_INTERFACE)
// extern "C"
// #endif
RustError::by_value mult_pippenger(point_t *out, const affine_t points[],
                                   size_t npoints, const scalar_t scalars[],
                                   size_t ffi_affine_sz)
{
    // return mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
    // printf("npoints: %d \n", npoints);
    // uint8_t *p1 = (uint8_t *)(uint64_t *)(&scalars[1]);
    // printf("first scalar x \n");
    // for(int i=0;i<32;i++) {
    //     printf("%x", p1[i]);
    // }
    // printf("\n");
    RustError r = mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);

    // fp_t *x = &out->X;
    // uint8_t *p = (uint8_t *)(uint64_t *)(limb_t *)x;
    // printf("result point x \n");
    // for(int i=0;i<32;i++) {
    //     printf("%x", p[i]);
    // }
    // printf("\n");
    return r;
}


#include <primitives/field.cuh>
#include <ff/bn254_params.cuh>
typedef Field<PARAMS_BN254::fp_config> g2_scalar_t;
RustError::by_value mult_pippenger_g2(point_t *out, const affine_t points[],
                                   size_t npoints, const scalar_t scalars[],
                                   size_t ffi_affine_sz)
{

}

// #endif
