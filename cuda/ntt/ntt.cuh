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
    // TODO: to remove this hard code, 16 is enough for most machines
    // TODO: add a method to drop the memory, as it is not auto dropped
    static std::map<uint32_t,  fr_t *> all_gpus_twiddle_forward_arr[16];
    static std::map<uint32_t,  fr_t *> all_gpus_twiddle_inverse_arr[16];

#ifndef __CUDA_ARCH__
    using namespace Ntt_Types;

    const uint32_t MAX_NUM_THREADS = 512;
    const uint32_t MAX_THREADS_BATCH = 512;          // TODO: allows 100% occupancy for scalar NTT for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;

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
     * Bit-reverses a batch of input arrays in-place inside GPU.
     * for example: on input array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
     * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
     * @param arr batch of arrays of some object of type T. Should be on GPU.
     * @param n length of `arr`.
     * @param logn log(n).
     * @param batch_size the size of the batch.
     */
    void reverse_order_batch(fr_t *arr, fr_t *arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size, stream_t &stream)
    {
        int number_of_threads = MAX_THREADS_BATCH;
        int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
        reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr, arr_reversed, n, logn, batch_size);
    }

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

    /**
     * @brief calculate twiddle factors
     */
    void fill_twiddle_factors_array(fr_t *d_twiddles, uint32_t n_twiddles, fr_t omega, stream_t &stream)
    {
        size_t size_twiddles = n_twiddles * sizeof(fr_t);
        twiddle_factors_kernel<<<1, 1, 0, stream>>>(d_twiddles, n_twiddles, omega);
        return;
    }

    /**
     * NTT/INTT inplace batch
     * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
     * @param d_inout Array for inplace processing
     * @param d_twiddles
     * @param n Length of `d_twiddles` array
     * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
     * @param inverse true for iNTT
     * @param is_coset true for multiplication by coset
     * @param coset should be array of length n - or in case of lesser than n, right-padded with zeroes
     * @param stream CUDA stream
     */
    void ntt_inplace_batch_template(
        fr_t *d_inout,
        fr_t *d_twiddles,
        unsigned n,
        unsigned batch_size,
        bool inverse,
        bool is_coset,
        fr_t *coset,
        stream_t &stream)
    {
        const int logn = int(log(n) / log(2));
        bool is_shared_mem_enabled = sizeof(fr_t) <= MAX_SHARED_MEM_ELEMENT_SIZE;
        const int log2_shmem_elems = is_shared_mem_enabled ? int(log(int(MAX_SHARED_MEM / sizeof(fr_t))) / log(2)) : logn;
        int num_threads = max(min(min(n / 2, MAX_THREADS_BATCH), 1 << (log2_shmem_elems - 1)), 1);
        const int chunks = max(int((n / 2) / num_threads), 1);
        const int total_tasks = batch_size * chunks;
        int num_blocks = total_tasks;
        const int shared_mem = 2 * num_threads * sizeof(fr_t); // TODO: calculator, as shared mem size may be more efficient less
                                                               // then max to allow more concurrent blocks on SM
        const int logn_shmem = is_shared_mem_enabled ? int(log(2 * num_threads) / log(2))
                                                     : 0; // TODO: shared memory support only for types <= 32 bytes

        if (inverse)
        {
            if (is_shared_mem_enabled)
                ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
                    d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem);

            for (int s = logn_shmem; s < logn; s++)
            {
                ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, false);
            }

            if (is_coset)
            {
                batch_vector_mult(coset, d_inout, n, batch_size, stream);
            }

            num_threads = max(min(n / 2, MAX_NUM_THREADS), 1);
            num_blocks = (n * batch_size + num_threads - 1) / num_threads;
            template_normalize_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n * batch_size, fr_t::inv_log_size(logn));
        }
        else
        {
            if (is_coset)
                batch_vector_mult(coset, d_inout, n, batch_size, stream);

            // printf("invoking ntt_template_kernel gpu_id: %d, is_shared_mem_enabled: %d >>>>>>>\n", stream.gpu_id, is_shared_mem_enabled);
            for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
            {
                ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, true);
            }

            if (is_shared_mem_enabled)
            {
                // printf("invoking ntt_template_kernel_shared_rev with gpu_id: %d >>>>>>>>\n", stream.gpu_id);
                ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
                    d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem, stream.gpu_id);
            }
        }
        // cudaStreamSynchronize(stream);
        return;
    }

    RustError InitTwiddleFactors(const gpu_t &gpu, size_t lg_domain_size)
    {
        gpu.select();
        // printf("init twiddles on device: %d\n", gpu.id());
         std::map<uint32_t, fr_t *> twiddle_p_map_forward;
         std::map<uint32_t, fr_t *> twiddle_p_map_inverse;

        all_gpus_twiddle_forward_arr[gpu.id()] = twiddle_p_map_forward;
        all_gpus_twiddle_inverse_arr[gpu.id()] = twiddle_p_map_inverse;
        // std::map<uint32_t, fr_t *> twiddle_p_map_inverse = all_gpus_twiddle_inverse_arr[gpu.id()];
        auto it = all_gpus_twiddle_forward_arr[gpu.id()].find(lg_domain_size);

        if (it == all_gpus_twiddle_forward_arr[gpu.id()].end())
        {

            size_t size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> twiddles_forward{size, gpu, true};
            dev_ptr_t<fr_t> twiddles_inverse{size, gpu, true};
            fill_twiddle_factors_array(&twiddles_forward[0], size, fr_t::omega(lg_domain_size), gpu);
            all_gpus_twiddle_forward_arr[gpu.id()][lg_domain_size] = twiddles_forward;
            fill_twiddle_factors_array(&twiddles_inverse[0], size, fr_t::omega_inv(lg_domain_size), gpu);
            all_gpus_twiddle_inverse_arr[gpu.id()][lg_domain_size] = twiddles_inverse;
        }

        gpu.sync();

        return RustError{cudaSuccess};
    }
    /**
     * \param gpu, which gpu to use, default is 0
     * \param inout, input and output fr array
     * \param lg_domain_size 2^{lg_domain_size} = N, where N is size of input array
     * \param order, specify the input output order (N: natural order, R: reversed order, default is NN)
     * \param direction, direction of NTT, farward, or inverse, default is farward
     * \param type, standard or coset, standard is the standard NTT, coset is the evaluation of shifted domain, default is standard
     * \param coset_ext_pow coset_ext_pow
     */
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

    /**
     * \param gpu, which gpu to use, default is 0
     * \param inout, input and output fr array
     * \param lg_domain_size 2^{lg_domain_size} = N, where N is size of input array
     * \param batches, The number of NTT batches to compute. Default value: 1.
     * \param order, specify the input output order (N: natural order, R: reversed order, default is NN)
     * \param direction, direction of NTT, farward, or inverse, default is farward
     * \param type, standard or coset, standard is the standard NTT, coset is the evaluation of shifted domain, default is standard
     * \param coset_ext_pow coset_ext_pow
     * \param are_outputs_on_device
     */
    // static
    RustError Batch(const gpu_t &gpu, fr_t *inout, uint32_t lg_domain_size, uint32_t batches,
                    InputOutputOrder order, Direction direction,
                    Type type, bool coset_ext_pow = false, bool are_outputs_on_device = false)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};



        try
        {
            gpu.select();

            size_t size = (size_t)1 << lg_domain_size;

            uint32_t n_twiddles = size;

            fr_t *d_twiddle;
            if (direction == Direction::inverse)
            {
                d_twiddle = all_gpus_twiddle_inverse_arr[gpu.id()].at(lg_domain_size);
            }
            else
            {
                d_twiddle = all_gpus_twiddle_forward_arr[gpu.id()].at(lg_domain_size);
            }

            size_t total_elements = size * batches;
            int input_size_bytes = total_elements * sizeof(fr_t);

            dev_ptr_t<fr_t> d_input{total_elements, gpu};
            dev_ptr_t<fr_t> d_input_reversed_tmp{total_elements, gpu};

            gpu.HtoD(&d_input[0], inout, total_elements);
            if (direction == Direction::inverse)
            {
                reverse_order_batch(d_input, &d_input_reversed_tmp[0], size, lg_domain_size, batches, gpu);
            }
            ntt_inplace_batch_template(direction == Direction::inverse ? d_input_reversed_tmp : d_input, d_twiddle, n_twiddles, batches, direction == Direction::inverse, false, nullptr, gpu);
            if (direction == Direction::forward)
            {
                reverse_order_batch(d_input, &d_input_reversed_tmp[0], size, lg_domain_size, batches, gpu);
            }
            if (!are_outputs_on_device)
            {
                // gpu.DtoH(inout, d_input, total_elements);
                gpu.DtoH(inout, &d_input_reversed_tmp[0], total_elements); // TODO: will cause illegal memory during concurrent batch ntt
                // cudaMemcpyAsync(inout, &d_input_reversed_tmp[0], input_size_bytes, cudaMemcpyDeviceToHost, gpu);
            }
            // gpu.select();
            gpu.sync(); // TODO: will cause illegal memory during concurrent batch ntt
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
#endif
}
#endif