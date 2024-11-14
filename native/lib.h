// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_CUDA_LIB_H__
#define __ZEKNOX_CUDA_LIB_H__

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif
#include <stdio.h>
#include <utils/rusterror.h>
#include <ntt/ntt.h>
#ifdef BUILD_MSM
#include <msm/msm.h>
#endif
#include <merkle/merkle.h>

EXTERN RustError get_number_of_gpus(size_t *ngpus);

EXTERN RustError list_devices_info();

EXTERN void init_cuda();

EXTERN void init_cuda_degree(const uint32_t max_degree);

EXTERN RustError init_twiddle_factors(size_t device_id, size_t lg_n);

EXTERN RustError init_coset(size_t device_id, size_t lg_domain_size, const uint64_t coset_gen);

EXTERN RustError compute_batched_ntt(size_t device_id, void *inout, uint32_t lg_domain_size,
                                     NTT_Direction ntt_direction,
                                     NTT_Config cfg);

EXTERN RustError compute_batched_lde(size_t device_id, void *output, void *input, uint32_t lg_domain_size,
                                     NTT_Direction ntt_direction,
                                     NTT_Config cfg);

EXTERN RustError compute_batched_lde_multi_gpu(void *output, void *input, uint32_t num_gpu, NTT_Direction ntt_direction,
                                               NTT_Config cfg,
                                               uint32_t lg_domain_size);

EXTERN RustError compute_transpose_rev(size_t device_id, void *output, void *input, uint32_t lg_n,
                                       NTT_TransposeConfig cfg);


#endif // __ZEKNOX_CUDA_LIB_H__
