#pragma once
# include <ff/alt_bn254.hpp>
# include <util/rusterror.h>
# include <ntt/ntt.h>

void bn128_add(fp_t *result, fp_t *a, fp_t *b);
RustError compute_ntt(size_t device_id, fr_t *inout, uint32_t lg_domain_size,
                Ntt_Types::InputOutputOrder ntt_order,
                Ntt_Types::Direction ntt_direction,
                Ntt_Types::Type ntt_type);

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
// typedef bucket_t::affine_inf_t affine_inf_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
RustError::by_value mult_pippenger(point_t* out, const affine_t points[],
                                       size_t npoints, const scalar_t scalars[],
                                       size_t ffi_affine_sz);