// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <ntt/ntt.h>
#include <ntt/ntt.cuh>
#include <ntt/kernels.cu>
#include <utils/batch_mul.cuh>
#include <cassert>
#include <map>
#include <vector>
#include <array>
#include <iostream>

#define TRANSPOSE_BLOCK_DIM 32
#define MAX_NUM_OF_GPUS 16

namespace ntt {

#ifndef __CUDA_ARCH__
    /**
     * TODO(cliff0412): add a method to drop the memory,
     * as it is not auto dropped
    */
    static std::array<std::array<fr_t *, 32>, MAX_NUM_OF_GPUS> all_gpus_twiddle_forward_arr;
    static std::array<std::array<fr_t *, 32>, MAX_NUM_OF_GPUS> all_gpus_twiddle_inverse_arr;
    static std::array<fr_t *, MAX_NUM_OF_GPUS> coset_ptr_arr;

    const uint32_t MAX_NUM_THREADS = 512;
    // TODO(cliff0412): allows 100% occupancy for scalar NTT for sm_86..sm_89
    const uint32_t MAX_THREADS_BATCH = 512;
    // TODO(cliff0412): occupancy calculator, hardcoded for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32;
    const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;

    /**
     * Bit-reverses a batch of input arrays in-place inside GPU.
     * for example: on input array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
     * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 switch places).
     * @param arr batch of arrays of some object of type T. Should be on GPU.
     * @param n length of `arr`.
     * @param logn log(n).
     * @param batch_size the size of the batch.
     */
    inline void reverse_order_batch(fr_t *arr, uint32_t n, uint32_t logn, uint32_t batch_size, stream_t &stream) {
        int number_of_threads = MAX_THREADS_BATCH;
        int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
        reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr, n, logn, batch_size);
    }

    inline void gen_random_salt(fr_t *arr, uint32_t chunk_size, uint32_t salt_size, uint64_t seed, stream_t &stream) {
        int number_of_threads = 1024;
        int number_of_blocks = (chunk_size + number_of_threads - 1) / number_of_threads;
        gen_random_salt_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr, chunk_size, salt_size, seed);
    }

    /**
     * Transposes a matrix into a new matrix
     * @param in_arr batch of input arrays of some object of type T. Should be on GPU.
     * @param out_arr batch of out arrays of some object of type T. Should be on GPU.
     * @param n length of `arr`.
     * @param batch_size the size of the batch.
     */
    inline void transpose_rev_batch(fr_t *in_arr, fr_t *out_arr, uint32_t n, uint32_t lg_n, uint32_t batch_size, stream_t &stream) {
        // This is the dimensions of the block, it is 64 rows and 8 cols however since each thread
        // transposes 8 elements, we consider the block size to be 64 x 64
        int blocks_per_row = (n + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM;
        int blocks_per_col = (batch_size + TRANSPOSE_BLOCK_DIM - 1) / TRANSPOSE_BLOCK_DIM;

        // Number of threads is max_threads and we create our 2d thread dimensions with 64x8 threads
        // Which constraints to the max threads defined prior
        int number_of_threads = MAX_NUM_THREADS;
        dim3 threads_dim = dim3(8, 32);
        dim3 blocks_dim = dim3(blocks_per_row, blocks_per_col);
        transpose_rev_kernel<<<blocks_dim, threads_dim, BLOCK_DIM*(BLOCK_DIM + 1)*sizeof(fr_t), stream>>>(in_arr, out_arr, n, lg_n, batch_size);
    }

    inline void extend_inputs_batch(fr_t *output, fr_t *arr, uint32_t n, uint32_t logn, uint32_t extension_rate_bits, uint32_t batch_size, stream_t &stream) {
        int number_of_threads = MAX_THREADS_BATCH;
        size_t n_extend = static_cast<size_t>(1 << (logn + extension_rate_bits));
        int number_of_blocks = (n_extend * batch_size + number_of_threads - 1) / number_of_threads;
        degree_extension_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(output, arr, n, n_extend, batch_size);
    }

    /**
     * @brief calculate twiddle factors
     */
    void fill_twiddle_factors_array(fr_t *d_twiddles, uint32_t n_twiddles, fr_t omega, stream_t &stream) {
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
        stream_t &stream) {
        const int logn = static_cast<int>(log(n) / log(2));
        bool is_shared_mem_enabled = sizeof(fr_t) <= MAX_SHARED_MEM_ELEMENT_SIZE;
        const int log2_shmem_elems = is_shared_mem_enabled ? static_cast<int>(log(static_cast<int>(MAX_SHARED_MEM / sizeof(fr_t))) / log(2)) : logn;
        int num_threads = max(min(min(n / 2, MAX_THREADS_BATCH), 1 << (log2_shmem_elems - 1)), 1);
        const int chunks = max(static_cast<int>((n / 2) / num_threads), 1);
        const int total_tasks = batch_size * chunks;
        int num_blocks = total_tasks;
        // TODO(cliff0412): calculator, as shared mem size may be more efficient less
        // then max to allow more concurrent blocks on SM
        const int shared_mem = 2 * num_threads * sizeof(fr_t);
        const int logn_shmem = is_shared_mem_enabled ? static_cast<int>(log(2 * num_threads) / log(2))
                                                     : 0;  // TODO(cliff0412): shared memory support only for types <= 32 bytes

        if (inverse) {
            if (is_shared_mem_enabled)
                ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
                    d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem);

            for (int s = logn_shmem; s < logn; s++) {
                ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, false);
            }

            if (is_coset) {
                batch_vector_mult(coset, d_inout, n, batch_size, stream);
            }

            num_threads = max(min(n / 2, MAX_NUM_THREADS), 1);
            num_blocks = (n * batch_size + num_threads - 1) / num_threads;
            template_normalize_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n * batch_size, fr_t::inv_log_size(logn));
        } else {
            if (is_coset)
                batch_vector_mult(coset, d_inout, n, batch_size, stream);
            // TODO(cliff0412): this loop also can be unrolled
            for (int s = logn - 1; s >= logn_shmem; s--) {
                ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, true);
            }

            if (is_shared_mem_enabled) {
                ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
                    d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem, stream.gpu_id);
            }
        }
        return;
    }

    RustError init_twiddle_factors(const gpu_t &gpu, size_t lg_domain_size) {
        gpu.select();
        size_t size = static_cast<size_t>(1 << lg_domain_size);
        dev_ptr_t<fr_t> twiddles_forward{size, gpu, true, true};
        dev_ptr_t<fr_t> twiddles_inverse{size, gpu, true, true};
        fill_twiddle_factors_array(&twiddles_forward[0], size, fr_t::omega(lg_domain_size), gpu);
        all_gpus_twiddle_forward_arr[gpu.id()][lg_domain_size] = twiddles_forward;
        fill_twiddle_factors_array(&twiddles_inverse[0], size, fr_t::omega_inv(lg_domain_size), gpu);
        all_gpus_twiddle_inverse_arr[gpu.id()][lg_domain_size] = twiddles_inverse;

        gpu.sync();

        return RustError{cudaSuccess};
    }

    /**
     * Used for benchmarking and wrapping into rust
     * @param in_arr batch of input arrays of some object of type T. Should be on GPU.
     * @param out_arr batch of out arrays of some object of type T. Should be on GPU.
     * @param n length of `arr`.
     * @param batch_size the size of the batch.
     */
    RustError compute_transpose_rev(const gpu_t &gpu, fr_t *output, fr_t *input, uint32_t lg_n, NTT_TransposeConfig cfg) {
        try {
            size_t size = static_cast<size_t>(1 << lg_n);
            size_t total_elements = size * cfg.batches;

            dev_ptr_t<fr_t> d_input{
                total_elements,
                gpu,
                cfg.are_inputs_on_device ? false : true,  // if inputs are already on device, no need to alloc input memory
                true                                      // drop input pointer after transpose
            };

            if (cfg.are_inputs_on_device) {
                d_input.set_device_ptr(input);
            } else {
                gpu.HtoD(&d_input[0], input, total_elements);
            }

            dev_ptr_t<fr_t> d_transpose_output{
                total_elements,
                gpu,
                cfg.are_outputs_on_device ? false : true,
                cfg.are_outputs_on_device ? true : false};

            if (cfg.are_outputs_on_device) {
                d_transpose_output.set_device_ptr(output);
            }

            transpose_rev_batch(d_input, d_transpose_output, size, lg_n, cfg.batches, gpu);

            if (!cfg.are_outputs_on_device) {
                gpu.DtoH(output, &d_transpose_output[0], total_elements);
            }

            gpu.sync();
        }
        catch (const cuda_error &e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }
        return RustError{cudaSuccess};
    }

    RustError init_coset(const gpu_t &gpu, size_t lg_domain_size, fr_t coset_gen) {
        gpu.select();
        // printf("start init coset gpu id : %d\n", gpu.id());
        size_t size = static_cast<size_t>(1 << lg_domain_size);
        dev_ptr_t<fr_t> d_coset{size, gpu, true, true};
        fill_twiddle_factors_array(&d_coset[0], size, coset_gen, gpu);
        coset_ptr_arr[gpu.id()] = d_coset;
        //  printf("end init coset \n");
        gpu.sync();

        return RustError{cudaSuccess};
    }

    /**
     * assume without coset
     * \param gpu, which gpu to use, default is 0
     * \param inout, input and output fr array
     * \param lg_domain_size 2^{lg_domain_size} = N, where N is size of input array
     * \param batches, The number of NTT batches to compute. Default value: 1.
     * \param order, specify the input output order (N: natural order, R: reversed order, default is NN)
     * \param direction, direction of NTT, farward, or inverse, default is farward
     * \param type, standard or coset, standard is the standard NTT, coset is the evaluation of shifted domain, default is standard
     * \param are_outputs_on_device
     */
    // static
    RustError batch_ntt(const gpu_t &gpu, fr_t *inout, uint32_t lg_domain_size, NTT_Direction direction, NTT_Config cfg) {
        // printf("inside batch ntt with coset: %d\n", cfg.with_coset);
        if (lg_domain_size == 0)
            return RustError{cudaErrorInvalidValue};

        try {
            gpu.select();

            size_t size = static_cast<size_t>(1 << lg_domain_size);
            uint32_t n_twiddles = size;

            fr_t *d_twiddle;
            if (direction == NTT_Direction::inverse) {
                d_twiddle = all_gpus_twiddle_inverse_arr[gpu.id()].at(lg_domain_size);
            } else {
                d_twiddle = all_gpus_twiddle_forward_arr[gpu.id()].at(lg_domain_size);
            }

            size_t total_elements = size * cfg.batches;

            dev_ptr_t<fr_t> d_input{
                total_elements,
                gpu,
                cfg.are_inputs_on_device ? false : true,  // if inputs are already on device, no need to alloc input memory
                cfg.are_outputs_on_device ? true : false  // if keep output on device; let the user drop the pointer
            };

            if (cfg.are_inputs_on_device) {
                d_input.set_device_ptr(inout);
            } else {
                gpu.HtoD(&d_input[0], inout, total_elements);
            }

            if (direction == NTT_Direction::inverse) {
                reverse_order_batch(d_input, size, lg_domain_size, cfg.batches, gpu);
            }
            ntt_inplace_batch_template(d_input, d_twiddle, n_twiddles, cfg.batches, direction == NTT_Direction::inverse, cfg.with_coset, coset_ptr_arr[gpu.id()], gpu);
            if (direction == NTT_Direction::forward) {
                reverse_order_batch(d_input, size, lg_domain_size, cfg.batches, gpu);
            }

            if (!cfg.are_outputs_on_device) {
                gpu.DtoH(inout, &d_input[0], total_elements);
            }

            gpu.sync();
        }
        catch (const cuda_error &e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    /**
     * assume with coset and that buffer has already been allocated in GPU with id 0
     * \param lg_n , logn before extension
     */
    RustError batch_lde_multi_gpu(fr_t *output, fr_t *inputs, size_t num_gpu, NTT_Direction direction, NTT_Config cfg, size_t lg_n, size_t total_num_input_elements, size_t total_num_output_elements) {
        if (total_num_input_elements == 0 || lg_n == 0 || cfg.extension_rate_bits < 1) {
            // printf("invalid input : %d\n", cfg.with_coset);
            return RustError{cudaErrorInvalidValue};
        }

        try {
            // printf("Multi-GPU LDE-starting\n");

            uint32_t num_batches_per_gpu = cfg.batches / num_gpu;
            uint32_t rem = cfg.batches % num_gpu;

            std::vector<fr_t *> output_pointers;
            std::vector<fr_t *> input_pointers;
            std::vector<uint32_t> batches_alloc;
            std::vector<uint32_t> batches_till;
            uint32_t batches_sum = 0;
            for (size_t i = 0; i < num_gpu; i++) {
                size_t b = (i < rem) ? num_batches_per_gpu + 1 : num_batches_per_gpu;
                batches_alloc.push_back(b);
                batches_till.push_back(batches_sum);
                batches_sum += b;
            }

            for (size_t i = 0; i < num_gpu; i++) {
                size_t batches = batches_alloc.at(i);
                if (batches == 0) {
                    continue;
                }
                auto &gpu = select_gpu(i);

                // printf("Num batches:%d\n", batches);
                uint32_t lg_output_domain_size = lg_n + cfg.extension_rate_bits;
                size_t size = static_cast<size_t>(1 << lg_output_domain_size);
                uint32_t n_twiddles = size;

                fr_t *d_twiddle;
                if (direction == NTT_Direction::inverse) {
                    d_twiddle = all_gpus_twiddle_inverse_arr[i].at(lg_output_domain_size);
                } else {
                    d_twiddle = all_gpus_twiddle_forward_arr[i].at(lg_output_domain_size);
                }

                size_t total_input_elements = (static_cast<size_t>(1 << lg_n)) * batches;

                // printf("Allocating input memory %ld B (log %ld, batches %ld) on GPU %d\n", total_input_elements, lg_n, batches, gpu.id());

                dev_ptr_t<fr_t> input_data{
                    total_input_elements,
                    gpu,
                    true,
                    true};

                void *src = (inputs + (batches_till.at(i) * (static_cast<size_t>(1 << lg_n))));
                gpu.HtoD(&input_data[0], src, total_input_elements);

                input_pointers.emplace_back(&input_data[0]);

                size_t total_output_elements = size * batches;

                // printf("Allocating output memory %ld B on GPU %d\n", total_output_elements, gpu.id());

                dev_ptr_t<fr_t> output_data{
                    total_output_elements,
                    gpu,
                    true,
                    true};

                output_pointers.emplace_back(&output_data[0]);

                extend_inputs_batch(&output_data[0], &input_data[0], static_cast<size_t>(1 << lg_n), lg_n, cfg.extension_rate_bits, batches, gpu);
                // gpu.sync();

                if (direction == NTT_Direction::inverse) {
                    reverse_order_batch(&output_data[0], size, lg_output_domain_size, batches, gpu);
                }

                ntt_inplace_batch_template(&output_data[0], d_twiddle, n_twiddles, batches, direction == NTT_Direction::inverse, cfg.with_coset, coset_ptr_arr[i], gpu);

                if (direction == NTT_Direction::forward) {
                    reverse_order_batch(&output_data[0], size, lg_output_domain_size, batches, gpu);
                }
            }

            auto &gpu = select_gpu(0);

            // printf("Allocating buffer memory %ld B on GPU %d\n", total_num_output_elements, gpu.id());

            dev_ptr_t<fr_t> d_buffer{
                total_num_output_elements,
                gpu,
                cfg.are_outputs_on_device ? false : true,
                cfg.are_outputs_on_device ? true : false};

            // gpu.sync();

            d_buffer.set_device_ptr(output);

            for (size_t i = 0; i < num_gpu; i++) {
                size_t batches = batches_alloc.at(i);
                if (batches == 0) {
                    continue;
                }

                // printf("Multi-GPU memory movement starting \n");
                auto &gpu = select_gpu(i);

                fr_t *output_data = output_pointers.at(i);
                // printf("Num batches:%d on GPU: %d\n", batches, gpu.id());
                uint32_t lg_output_domain_size = lg_n + cfg.extension_rate_bits;
                size_t size = static_cast<size_t>(1 << lg_output_domain_size);
                size_t total_output_elements = size * batches;
                size_t total_output_bytes = total_output_elements * sizeof(fr_t);
                // printf("Bytes:%d on GPU: %d\n", total_output_bytes, gpu.id());

                if (i == 0) {
                    // printf("Offset:0 on GPU: %d\n", gpu.id());
                    CUDA_OK(cudaMemcpyAsync(&d_buffer[0], output_data, total_output_bytes, cudaMemcpyDeviceToDevice, gpu));
                    // printf("GPU 0 memory moved\n");
                } else {
                    int canAccessPeer = 0;
                    CUDA_OK(cudaDeviceCanAccessPeer(&canAccessPeer, 0, gpu.id()));
                    if (canAccessPeer) {
                        // printf("Peer copy can access gpu\n");
                        size_t offset = batches_till.at(i) * (static_cast<size_t>(1 << lg_output_domain_size));
                        // printf("Offset:%d on GPU: %d\n", offset, gpu.id());
                        // printf("The pointer value: %d\n", &d_buffer[offset]);
                        CUDA_OK(cudaMemcpyPeerAsync(&d_buffer[offset], 0, output_data, gpu.id(), total_output_bytes, gpu));
                    }
                }
            }

            for (size_t i = 0; i < num_gpu; i++) {
                size_t batches = batches_alloc.at(i);
                if (batches == 0) {
                    continue;
                }
                auto &gpu = select_gpu(i);
                fr_t *output_data = output_pointers.at(i);
                fr_t *input_data = input_pointers.at(i);
                // printf("Syncing gpu %d\n", gpu.id());
                gpu.sync();
                CUDA_OK(cudaFree(reinterpret_cast<void *>(output_data)));
                CUDA_OK(cudaFree(reinterpret_cast<void *>(input_data)));
            }
        }
        catch (const cuda_error &e) {
            printf("Err %d\n", e.code());
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    /**
     * assume with coset
     * \param lg_n , logn before extension
     */
    RustError batch_lde(const gpu_t &gpu, fr_t *output, fr_t *input, uint32_t lg_n, NTT_Direction direction, NTT_Config cfg) {
        if (lg_n == 0 || cfg.extension_rate_bits < 1) {
            // printf("invalid input : %d\n", cfg.with_coset);
            return RustError{cudaErrorInvalidValue, "invalid args"};
        }

        try {
            gpu.select();
            // printf("batch lde with input lg_n:%d,  extension_rate_bits: %d, direction: %d, batches: %d, are_outputs_on_device: %d, are_inputs_on_device: %d\n", lg_n, cfg.extension_rate_bits, direction,
            //        cfg.batches,
            //        cfg.are_outputs_on_device, cfg.are_inputs_on_device);

            uint32_t lg_output_domain_size = lg_n + cfg.extension_rate_bits;
            size_t size = static_cast<size_t>(1 << lg_output_domain_size);
            uint32_t n_twiddles = size;

            fr_t *d_twiddle;
            if (direction == NTT_Direction::inverse) {
                d_twiddle = all_gpus_twiddle_inverse_arr[gpu.id()].at(lg_output_domain_size);
            } else {
                d_twiddle = all_gpus_twiddle_forward_arr[gpu.id()].at(lg_output_domain_size);
            }

            size_t total_input_elements = (static_cast<size_t>(1 << lg_n)) * cfg.batches;
            int input_size_bytes = total_input_elements * sizeof(fr_t);

            dev_ptr_t<fr_t> d_input{
                total_input_elements,
                gpu,
                cfg.are_inputs_on_device ? false : true,  // new device input has to be allocated
                cfg.are_inputs_on_device ? true : false   // if keep output on device; let the user drop the pointer
            };

            if (cfg.are_inputs_on_device) {
                // printf("set input device pointer: %x\n", input);
                d_input.set_device_ptr(input);
            } else {
                gpu.HtoD(&d_input[0], input, total_input_elements);
            }

            size_t total_output_elements = size * (cfg.batches + cfg.salt_size);
            int input_output_bytes = total_output_elements * sizeof(fr_t);
            dev_ptr_t<fr_t> d_output{
                total_output_elements,
                gpu,
                cfg.are_outputs_on_device ? false : true,
                cfg.are_outputs_on_device ? true : false};

            if (cfg.are_outputs_on_device) {
                d_output.set_device_ptr(output);
            }

            extend_inputs_batch(&d_output[0], &d_input[0], static_cast<size_t>(1 << lg_n), lg_n, cfg.extension_rate_bits, cfg.batches, gpu);

            if (direction == NTT_Direction::inverse) {
                reverse_order_batch(d_output, size, lg_output_domain_size, cfg.batches, gpu);
            }
            // printf("start inplace batch template, with coset: %d \n", cfg.with_coset);
            ntt_inplace_batch_template(d_output, d_twiddle, n_twiddles, cfg.batches, direction == NTT_Direction::inverse, cfg.with_coset, coset_ptr_arr[gpu.id()], gpu);
            // printf("end inplace batch template, with coset: %d \n", cfg.with_coset);
            if (direction == NTT_Direction::forward) {
                reverse_order_batch(d_output, size, lg_output_domain_size, cfg.batches, gpu);
            }

#if defined(FEATURE_GOLDILOCKS)
            if (cfg.salt_size > 0) {
                uint64_t seed = time(NULL);
                gen_random_salt(&d_output[size * cfg.batches], size, cfg.salt_size, seed, gpu);
            }
#endif

            if (!cfg.are_outputs_on_device) {
                // printf("start copy device to host \n");
                gpu.DtoH(output, &d_output[0], total_output_elements);
            }

            gpu.sync();
        }
        catch (const cuda_error &e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }
#endif
}  // namespace ntt
