/** macro utils*/
#define XSTR(x) STR(x)
#define STR(x) #x

#include <cuda.h>

#ifdef NDEBUG
#define CUDA_DEBUG false
#else
#define CUDA_DEBUG true
#include <cstdio>
#endif


#pragma message "The value of CUDA_DEBUG: " XSTR(CUDA_DEBUG)
#pragma message "The value of __CUDA_ARCH__: " XSTR(__CUDA_ARCH__)

#if defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn254.hpp>
#else
#error "no FEATURE"
#endif

#include "lib.h"
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

#include <msm/pippenger.cuh>
#include <cstdio>
#include <blst_t.hpp>

RustError::by_value mult_pippenger(point_t *result, const affine_t points[],
                                   size_t npoints, const scalar_t scalars[],
                                   size_t ffi_affine_sz)
{
    // return mult_pippenger<bucket_t>(result, points, npoints, scalars, false, ffi_affine_sz);
    // printf("npoints: %d \n", npoints);
    // uint8_t *p1 = (uint8_t *)(uint64_t *)(&scalars[1]);
    // printf("first scalar x \n");
    // for(int i=0;i<32;i++) {
    //     printf("%x", p1[i]);
    // }
    // printf("\n");
    RustError r = mult_pippenger<bucket_t>(result, points, npoints, scalars, false, ffi_affine_sz);

    // fp_t *x = &result->X;
    // uint8_t *p = (uint8_t *)(uint64_t *)(limb_t *)x;
    // printf("result point x \n");
    // for(int i=0;i<32;i++) {
    //     printf("%x", p[i]);
    // }
    // printf("\n");
    return r;
}

#if defined(FEATURE_BN254)
extern "C"
RustError::by_value mult_pippenger_g2(g2_projective_t* result, g2_affine_t* points, size_t msm_size, scalar_field_t* scalars, size_t large_bucket_factor, bool on_device, bool big_triangle)
{
    large_msm<scalar_field_t, g2_projective_t, g2_affine_t>(
        scalars, points, msm_size, result, on_device, big_triangle, large_bucket_factor);

    // std::cout << "after copy result" << std::endl;
    // uint32_t* out_x_real = (*result).x.real.export_limbs();
    // printf("output, x.real %d \n", *(out_x_real) );


}
#endif
