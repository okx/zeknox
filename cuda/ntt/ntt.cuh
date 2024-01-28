// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO_NTT_NTT_CUH__
#define __CRYPTO_NTT_NTT_CUH__

#include <cassert>
#include <util/gpu_t.cuh>
#include <util/rusterror.h>
#include <util/batch_mul.cuh>
#include <ntt/ntt.h>
#include "parameters.cuh"
#include "kernels.cu"
#include <map>
#include <vector>

namespace ntt
{

    static int max_size;
    static fr_t *twiddles;
    static std::map<fr_t, int> coset_index_map;

#ifndef __CUDA_ARCH__
    using namespace Ntt_Types;

    const uint32_t MAX_NUM_THREADS = 512;            // TODO: hotfix - should be 1024, currently limits shared memory size
    const uint32_t MAX_THREADS_BATCH = 512;          // TODO: allows 100% occupancy for scalar NTT for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;



    // class NTT
    // {

    // protected:
    // static
    void bit_rev(fr_t *d_out, const fr_t *d_inp, uint32_t lg_domain_size, stream_t &stream)
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;
        // aim to read 4 cache lines of consecutive data per read
        const size_t Z_COUNT = 256 / sizeof(fr_t); // 32 for goldilocks

        if (domain_size <= WARP_SZ)
            bit_rev_permutation<<<1, domain_size, 0, stream>>>(d_out, d_inp, lg_domain_size);
        else if (d_out == d_inp || domain_size <= Z_COUNT * Z_COUNT)
            bit_rev_permutation<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>(d_out, d_inp, lg_domain_size);
        else if (domain_size < 128 * Z_COUNT)
            bit_rev_permutation_aux<<<1, domain_size / Z_COUNT, domain_size * sizeof(fr_t), stream>>>(d_out, d_inp, lg_domain_size);
        else
            bit_rev_permutation_aux<<<domain_size / Z_COUNT / 128, 128, Z_COUNT * 128 * sizeof(fr_t),
                                      stream>>>(d_out, d_inp, lg_domain_size); // Z_COUNT * sizeof(fr_t) is 256 bytes

        CUDA_OK(cudaGetLastError());
    }

    /**
     * @brief reverse order for batched inputs
     * @param[in] arr_in, inputs
     * @param[in] n, array size per batch
     * @param[in] logn, 2^{logn} =n
     * @param[in] batch_size, number of batches
     */
    // static
    void reverse_order_batch(fr_t *arr_in, uint32_t n, uint32_t logn, uint32_t batch_size, stream_t &stream, fr_t *arr_out)
    {
        int number_of_threads = MAX_THREADS_BATCH;
        int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
        reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr_in, arr_out, n, logn, batch_size);
    }

    /**
     * NTT/INTT inplace batch
     * Note: this function does not perform any bit-reverse permutations on its inputs or outputs.
     * @param d_input Input array
     * @param n Size of `d_input`
     * @param d_twiddles Twiddles
     * @param n_twiddles Size of `d_twiddles`
     * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
     * @param inverse true for iNTT
     * @param coset should be array of lenght n or a nullptr if NTT is not computed on a coset
     * @param stream CUDA stream
     * @param is_async if false, perform sync of the supplied CUDA stream at the end of processing
     * @param d_output Output array
     */
    // static
    void ntt_inplace_batch_template(
        fr_t *d_input,
        int n,
        fr_t *d_twiddles,
        int n_twiddles,
        int batch_size,
        int logn,
        bool inverse,
        bool ct_buttterfly,
        fr_t *arbitrary_coset,
        int coset_gen_index,
        stream_t &stream,
        fr_t *d_output)
    {

        bool is_shared_mem_enabled = sizeof(fr_t) <= MAX_SHARED_MEM_ELEMENT_SIZE;
        const int log2_shmem_elems = is_shared_mem_enabled ? int(log(int(MAX_SHARED_MEM / sizeof(fr_t))) / log(2)) : logn;
        int num_threads = max(min(min(n / 2, MAX_THREADS_BATCH), 1 << (log2_shmem_elems - 1)), 1);
        const int chunks = max(int((n / 2) / num_threads), 1);
        const int total_tasks = batch_size * chunks;
        int num_blocks = total_tasks;
        const int shared_mem = 2 * num_threads * sizeof(fr_t); // TODO: calculator, as shared mem size may be more efficient
                                                               // less then max to allow more concurrent blocks on SM
        const int logn_shmem = is_shared_mem_enabled ? int(log(2 * num_threads) / log(2))
                                                     : 0; // TODO: shared memory support only for types <= 32 bytes
        int num_threads_coset = max(min(n / 2, MAX_NUM_THREADS), 1);
        int num_blocks_coset = (n * batch_size + num_threads_coset - 1) / num_threads_coset;

        if (inverse)
        {
            d_twiddles = d_twiddles + n_twiddles;
            n_twiddles = -n_twiddles;
        }

        bool is_on_coset = (coset_gen_index != 0) || arbitrary_coset;
        bool direct_coset = (!inverse && is_on_coset);
        if (direct_coset)
            utils_internal::BatchMulKernel<fr_t, fr_t><<<num_blocks_coset, num_threads_coset, 0, stream>>>(
                d_input, n, batch_size, arbitrary_coset ? arbitrary_coset : d_twiddles, arbitrary_coset ? 1 : coset_gen_index,
                n_twiddles, logn, ct_buttterfly, d_output);

        if (ct_buttterfly)
        {
            if (is_shared_mem_enabled)
                ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
                    direct_coset ? d_output : d_input, 1 << logn_shmem, d_twiddles, n_twiddles, total_tasks, 0, logn_shmem,
                    d_output);

            for (int s = logn_shmem; s < logn; s++) // TODO: this loop also can be unrolled
            {
                ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(
                    (direct_coset || (s > 0)) ? d_output : d_input, n, d_twiddles, n_twiddles, total_tasks, s, false, d_output);
            }
        }
        else
        {
            for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
            {
                ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(
                    (direct_coset || (s < logn - 1)) ? d_output : d_input, n, d_twiddles, n_twiddles, total_tasks, s, true,
                    d_output);
            }

            if (is_shared_mem_enabled)
                ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
                    (direct_coset || (logn > logn_shmem)) ? d_output : d_input, 1 << logn_shmem, d_twiddles, n_twiddles,
                    total_tasks, 0, logn_shmem, d_output);
        }

        if (inverse)
        {
            if (is_on_coset)
                utils_internal::BatchMulKernel<fr_t, fr_t><<<num_blocks_coset, num_threads_coset, 0, stream>>>(
                    d_output, n, batch_size, arbitrary_coset ? arbitrary_coset : d_twiddles,
                    arbitrary_coset ? 1 : -coset_gen_index, -n_twiddles, logn, !ct_buttterfly, d_output);

            utils_internal::NormalizeKernel<fr_t, fr_t>
                <<<num_blocks_coset, num_threads_coset, 0, stream>>>(d_output, fr_t::inv_log_size(logn), n * batch_size);
        }

        return;
    }

    // static
    void NTT_internal(fr_t *d_inout, uint32_t lg_domain_size,
                      InputOutputOrder order, Direction direction,
                      Type type, stream_t &stream,
                      bool coset_ext_pow = false)
    {
        const bool intt = direction == Direction::inverse;
        const auto &ntt_parameters = *NTTParameters::all(intt)[stream];
        bool bitrev;
        Algorithm algorithm;

        switch (order)
        {
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
        switch (algorithm)
        {
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

    // public:
    /**
     * \param gpu, which gpu to use, default is 0
     * \param inout, input and output fr array
     * \param lg_domain_size 2^{lg_domain_size} = N, where N is size of input array
     * \param order, specify the input output order (N: natural order, R: reversed order, default is NN)
     * \param direction, direction of NTT, farward, or inverse, default is farward
     * \param type, standard or coset, standard is the standard NTT, coset is the evaluation of shifted domain, default is standard
     * \param coset_ext_pow coset_ext_pow
     */
    // static
    RustError Base(const gpu_t &gpu, fr_t *inout, uint32_t lg_domain_size,
                   InputOutputOrder order, Direction direction,
                   Type type, bool coset_ext_pow = false)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try
        {

            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_inout{domain_size, gpu};
            gpu.HtoD(&d_inout[0], inout, domain_size);
            NTT_internal(&d_inout[0], lg_domain_size, order, direction, type, gpu,
                         coset_ext_pow);
            gpu.DtoH(inout, &d_inout[0], domain_size);
            gpu.sync();
        }
        catch (const cuda_error &e)
        {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    // static
    void InitDomain(fr_t primitive_root, const gpu_t &gpu)
    {

        // only generate twiddles if they haven't been generated yet
        // please note that this is not thread-safe at all,
        // but it's a singleton that is supposed to be initialized once per program lifetime
        if (!twiddles)
        {
            fr_t omega = primitive_root;
            for (int i = 0; i < 32; i++) // TWO_ADICITY = 32
                omega = omega ^ 2;
            if (omega != fr_t::one())
            {
                std::cerr << "Primitive root provided to the InitDomain function is not in the subgroup" << '\n';
                throw -1;
            }

            std::vector<fr_t> h_twiddles;
            h_twiddles.push_back(fr_t::one());
            int n = 1;
            do
            {
                coset_index_map[h_twiddles.at(n - 1)] = n - 1;
                h_twiddles.push_back(h_twiddles.at(n - 1) * primitive_root);
            } while (h_twiddles.at(n++) != fr_t::one());

            cudaMallocAsync(&twiddles, n * sizeof(fr_t), gpu);
            cudaMemcpyAsync(twiddles, &h_twiddles.front(), n * sizeof(fr_t), cudaMemcpyHostToDevice, gpu);

            max_size = n - 1;
            cudaStreamSynchronize(gpu);
        }

        return;
    }

    /**
     * \param gpu, which gpu to use, default is 0
     * \param inout, input and output fr array
     * \param lg_domain_size 2^{lg_domain_size} = N, where N is size of input array
     * \param batch_size, The number of NTTs to compute. Default value: 1.
     * \param order, specify the input output order (N: natural order, R: reversed order, default is NN)
     * \param direction, direction of NTT, farward, or inverse, default is farward
     * \param type, standard or coset, standard is the standard NTT, coset is the evaluation of shifted domain, default is standard
     * \param coset_ext_pow coset_ext_pow
     * \param are_outputs_on_device
     */
    // static
    RustError Batch(const gpu_t &gpu, fr_t *inout, uint32_t lg_domain_size, uint32_t batch_size,
                    InputOutputOrder order, Direction direction,
                    Type type, bool coset_ext_pow = false, bool are_outputs_on_device = false)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try
        {

            gpu.select();

            size_t size = (size_t)1 << lg_domain_size;
            int input_size_bytes = size * batch_size * sizeof(fr_t);
            dev_ptr_t<fr_t> d_input{input_size_bytes, gpu};
            gpu.HtoD(&d_input[0], inout, input_size_bytes);

            dev_ptr_t<fr_t> d_output{input_size_bytes, gpu};

            fr_t *coset = nullptr;
            int coset_index = 0;
            try
            {
                coset_index = coset_index_map.at(fr_t::one());
            }
            catch (...)
            {
                // if coset index is not found in the subgroup, compute coset powers on CPU and move them to device
                std::vector<fr_t> h_coset;
                h_coset.push_back(fr_t::one());
                fr_t coset_gen = (direction == Direction::inverse) ? fr_t::one().reciprocal() : fr_t::one();
                for (int i = 1; i < size; i++)
                {
                    h_coset.push_back(h_coset.at(i - 1) * coset_gen);
                }
                CUDA_OK(cudaMallocAsync(&coset, size * sizeof(fr_t), gpu));
                CUDA_OK(cudaMemcpyAsync(coset, &h_coset.front(), size * sizeof(fr_t), cudaMemcpyHostToDevice, gpu));
                h_coset.clear();
            }

            bool ct_butterfly = true;
            bool reverse_input = false;
            switch (order)
            {
            case InputOutputOrder::NN:
                reverse_input = true;
                break;
            case InputOutputOrder::NR:
                ct_butterfly = false;
                break;
            case InputOutputOrder::RR:
                reverse_input = true;
                ct_butterfly = false;
                break;
            }
            printf("before reverse_order_batch, size: %d, batch_size: %d, reverse_input: %d\n", size, batch_size, reverse_input);
            if (reverse_input)
                reverse_order_batch(d_input, size, lg_domain_size, batch_size, gpu, d_output);

            ntt_inplace_batch_template(
                reverse_input ? d_output : d_input, size, twiddles, max_size, batch_size, lg_domain_size,
                direction == Direction::inverse, ct_butterfly, coset, coset_index, gpu, d_output);

  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":" << __LINE__ << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
            if (!are_outputs_on_device)
                CUDA_OK(cudaMemcpyAsync(inout, d_output, input_size_bytes, cudaMemcpyDeviceToHost, gpu));

            //     if (coset) CHK_IF_RETURN(cudaFreeAsync(coset, stream));
            //     if (!are_inputs_on_device) CHK_IF_RETURN(cudaFreeAsync(d_input, stream));
            //     if (!are_outputs_on_device) CHK_IF_RETURN(cudaFreeAsync(d_output, stream));
            //     if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));

            //     return CHK_LAST();

            //             NTT_internal(&d_inout[0], lg_domain_size, order, direction, type, gpu,
            //                          coset_ext_pow);
            //             gpu.DtoH(inout, &d_inout[0], domain_size);
            //             gpu.sync();
        }
        catch (const cuda_error &e)
        {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }
    // };

#endif
}
#endif