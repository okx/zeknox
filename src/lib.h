#pragma once
# include <ff/alt_bn128.hpp>
# include <util/rusterror.h>
# include <ntt/ntt.h>

void bn128_add(fp_t *result, fp_t *a, fp_t *b);
RustError compute_ntt(size_t device_id, fr_t *inout, uint32_t lg_domain_size,
                Ntt_Types::InputOutputOrder ntt_order,
                Ntt_Types::Direction ntt_direction,
                Ntt_Types::Type ntt_type);