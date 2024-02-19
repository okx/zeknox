#pragma once
#include <util/rusterror.h>
#include <ntt/ntt.h>
#include <ff/goldilocks.hpp>
#include <util/device_context.cuh>

#ifndef FEATURE_GOLDILOCKS
#include <ff/alt_bn254.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
RustError::by_value mult_pippenger(point_t *out, const affine_t points[],
                                   size_t npoints, const scalar_t scalars[],
                                   size_t ffi_affine_sz);

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
typedef Projective<g2_point_field_t, scalar_field_t, g2_b, g2_gen_x, g2_gen_y> g2_projective_t;
typedef Affine<g2_point_field_t> g2_affine_t;

extern "C" RustError::by_value mult_pippenger_g2(g2_projective_t *out, g2_affine_t *points, size_t msm_size, scalar_field_t *scalars, size_t large_bucket_factor, bool on_device,
                                                 bool big_triangle);
#endif

extern "C" RustError compute_ntt(size_t device_id, fr_t *inout, uint32_t lg_domain_size,
                                 Ntt_Types::InputOutputOrder ntt_order,
                                 Ntt_Types::Direction ntt_direction,
                                 Ntt_Types::Type ntt_type);

extern "C" RustError compute_batched_ntt(size_t device_id, fr_t *inout, uint32_t lg_domain_size,
                                         Ntt_Types::Direction ntt_direction,
                                         Ntt_Types::NTTConfig cfg);

extern "C" RustError compute_batched_lde(size_t device_id, fr_t *output, fr_t *input, uint32_t lg_domain_size,
                                         Ntt_Types::Direction ntt_direction,
                                         Ntt_Types::NTTConfig cfg);

extern "C" RustError init_twiddle_factors(size_t device_id, size_t lg_n);

extern "C" RustError init_coset(size_t device_id, size_t lg_domain_size, fr_t coset_gen);

extern "C" RustError get_number_of_gpus(size_t *ngpus);
