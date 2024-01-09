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

#include <msm/pippenger.cuh>
#include <cstdio>
#include <blst_t.hpp>

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

#if defined(FEATURE_BN254)

#include <primitives/field.cuh>
#include <primitives/extension_field.cuh>
#include <primitives/projective.cuh>
#include <ff/bn254_params.cuh>
typedef Field<PARAMS_BN254::fp_config> scalar_field_t;
typedef Field<PARAMS_BN254::fq_config> point_field_t;
typedef ExtensionField<PARAMS_BN254::fq_config> g2_point_field_t;
static constexpr g2_point_field_t g2_gen_x =
    g2_point_field_t{point_field_t{PARAMS_BN254::g2_gen_x_re}, point_field_t{PARAMS_BN254::g2_gen_x_im}};
static constexpr g2_point_field_t g2_gen_y =
    g2_point_field_t{point_field_t{PARAMS_BN254::g2_gen_y_re}, point_field_t{PARAMS_BN254::g2_gen_y_im}};
static constexpr g2_point_field_t g2_b = g2_point_field_t{
    point_field_t{PARAMS_BN254::weierstrass_b_g2_re}, point_field_t{PARAMS_BN254::weierstrass_b_g2_im}};
typedef Projective<g2_point_field_t, scalar_t, g2_b, g2_gen_x, g2_gen_y> g2_projective_t;
typedef Affine<g2_point_field_t> g2_affine_t;

RustError::by_value mult_pippenger_g2(g2_projective_t *out, g2_affine_t *points,
                                      size_t msm_size, const scalar_field_t *scalars,
                                      size_t large_bucket_factor)
{
    scalar_field_t *scalars_d;
    g2_affine_t *points_d;
    g2_projective_t *out_d;
    cudaMalloc(&scalars_d, sizeof(scalar_field_t) * msm_size);
    cudaMalloc(&points_d, sizeof(g2_affine_t) * msm_size);
    cudaMalloc(&out_d, sizeof(g2_projective_t));
    cudaMemcpy(scalars_d, scalars, sizeof(scalar_field_t) * msm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(points_d, points, sizeof(g2_affine_t) * msm_size, cudaMemcpyHostToDevice);
    std::cout << "finished copying" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pippenger_g2<scalar_field_t, g2_projective_t, g2_affine_t>(
        scalars_d, points_d, msm_size, out_d, true, false, large_bucket_factor, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    // std::cout << g2_projective_t::to_affine(out_d) << std::endl;

    cudaMemcpy(&out, out_d, sizeof(g2_projective_t), cudaMemcpyDeviceToHost);
}
#endif
