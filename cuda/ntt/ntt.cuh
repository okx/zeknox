// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO_NTT_NTT_CUH__
#define __CRYPTO_NTT_NTT_CUH__

#include <cassert>
#include <util/gpu_t.cuh>
#include <util/rusterror.h>
#include <ntt/ntt.h>
#include "parameters.cuh"
#include "kernels.cu"
#ifndef __CUDA_ARCH__
using namespace Ntt_Types;

class NTT {

protected:
    static void bit_rev(fr_t* d_out, const fr_t* d_inp, uint32_t lg_domain_size, stream_t& stream) 
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;
        // aim to read 4 cache lines of consecutive data per read
        const size_t Z_COUNT = 256 / sizeof(fr_t);  // 32 for goldilocks

        if (domain_size <= WARP_SZ)
            bit_rev_permutation
                <<<1, domain_size, 0, stream>>>
                (d_out, d_inp, lg_domain_size);
        else if (d_out == d_inp || domain_size <= Z_COUNT * Z_COUNT)
            bit_rev_permutation
                <<<domain_size/WARP_SZ, WARP_SZ, 0, stream>>>
                (d_out, d_inp, lg_domain_size);
        else if (domain_size < 128 * Z_COUNT)
            bit_rev_permutation_aux
                <<<1, domain_size / Z_COUNT, domain_size * sizeof(fr_t), stream>>>
                (d_out, d_inp, lg_domain_size);
        else
            bit_rev_permutation_aux
                <<<domain_size / Z_COUNT / 128, 128, Z_COUNT * 128 * sizeof(fr_t),  
                   stream>>>
                (d_out, d_inp, lg_domain_size);  // Z_COUNT * sizeof(fr_t) is 256 bytes

        CUDA_OK(cudaGetLastError());
    }

    static void NTT_internal(fr_t* d_inout, uint32_t lg_domain_size,
                             InputOutputOrder order, Direction direction,
                             Type type, stream_t& stream,
                             bool coset_ext_pow = false) 
    {
        const bool intt = direction == Direction::inverse;
        const auto& ntt_parameters = *NTTParameters::all(intt)[stream];
        bool bitrev;
        Algorithm algorithm;

        switch (order) {
            case InputOutputOrder::NN:
                bit_rev(d_inout, d_inout, lg_domain_size, stream);
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::NR:
                bitrev = false;
                algorithm = Algorithm::GS;
                break;
            case InputOutputOrder::RN:
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::RR:
                bitrev = true;
                algorithm = Algorithm::GS;
                break;
            default:
                assert(false);
        }
        // printf("inside NTT_internal \n");
        switch (algorithm) {
            case Algorithm::GS:
                // TODO:
                // GS_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
            case Algorithm::CT:
                CT_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
        }
        if (order == InputOutputOrder::RR)
            bit_rev(d_inout, d_inout, lg_domain_size, stream);
    }

public:
    /**
     * \param gpu, which gpu to use, default is 0
     * \param inout, input and output fr array
     * \param lg_domain_size 2^{lg_domain_size} = N, where N is size of input array
     * \param order, specify the input output order (N: natural order, R: reversed order, default is NN)
     * \param direction, direction of NTT, farward, or inverse, default is farward
     * \param type, standard or coset, standard is the standard NTT, coset is the evaluation of shifted domain, default is standard
     * \param coset_ext_pow coset_ext_pow
    */
    static RustError Base(const gpu_t& gpu, fr_t* inout, uint32_t lg_domain_size,
                          InputOutputOrder order, Direction direction,
                          Type type, bool coset_ext_pow = false)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};
        
        try {

            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_inout{domain_size, gpu};
            gpu.HtoD(&d_inout[0], inout, domain_size);
            NTT_internal(&d_inout[0], lg_domain_size, order, direction, type, gpu,
                         coset_ext_pow);
            gpu.DtoH(inout, &d_inout[0], domain_size);
            gpu.sync();

        } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};

    }

};

#endif
#endif