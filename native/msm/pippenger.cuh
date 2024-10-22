#ifndef __CRYPTO_MSM_PIPPENGER_CUH__
#define __CRYPTO_MSM_PIPPENGER_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cassert>

#include <utils/vec2d_t.hpp>
#include <utils/slice_t.hpp>

#include <utils/exception.cuh>
#include <msm/kernels/pippenger_kernels.cuh>

#ifndef WARP_SZ
#define WARP_SZ 32
#endif


#define MAX_TH 256
#define WINDOW_BIT_SIZE 16

#include <vector>
#include <utils/exception.cuh>
#include <utils/rusterror.h>
#include <utils/gpu_t.cuh>
#include <chrono>
#include <msm/msm.h>

/**
 * `P` result type
 * `A` point type
 * `S` scalar type
 */
template <typename P, typename A, typename S>
class msm_t
{
    const gpu_t &gpu;
    size_t npoints;
    uint32_t wbits, nwins;
// typedef g1_affine_t A;
// typedef scalar_field_t S;
public:
    A *d_points;
    S *d_scalars;

    constexpr static int lg2(size_t n)
    {
        int ret = 0;
        while (n >>= 1)
            ret++;
        return ret;
    }

public:
    msm_t(A *points, S *scalars, size_t np, int device_id = -1)
        : gpu(select_gpu(device_id)), d_points(points), d_scalars(scalars)
    {
        // std::cout << "Line: " << __LINE__ << "\n";
        // printf("bucket_t::degree: %d\n", bucket_t::degree);
        // npoints = (np + WARP_SZ - 1) & ((size_t)0 - WARP_SZ); // make npoints a multiple of WARP_SZ

        // confine the number of bits in each window in the ragne [10,18]
        wbits = 17;
        if (npoints > 192)
        {
            wbits = std::min(lg2(npoints), 18);
            if (wbits < 10)
                wbits = 10;
        }
        else if (npoints > 0)
        {
            wbits = 10;
        }
        nwins = (scalar_t::nbits - 1) / wbits + 1;
    }
    inline msm_t(int device_id = -1)
        : msm_t(nullptr, nullptr, 0, device_id) {
          };
    ~msm_t()
    {
        gpu.sync();
    }

    /**
     * this function computes msm using the bucket method
     * @param bitsize, bit size of scalar; e.g 254 for BN254 field;
     * @param on_device whetehr put result on device; default is `false`
     */
    void bucket_method_msm(
        P *final_result,
        A *points,
        S *scalars,
        unsigned size,
        bool mont,
        bool on_device,
        bool big_triangle,
        unsigned large_bucket_factor)
    {
        P *buckets;

        unsigned num_of_window = nwins; // number of windows
        unsigned msm_log_size = ceil(log2(size));
        unsigned num_of_windows_bitsize = ceil(log2(num_of_window));
        unsigned total_num_of_buckets = num_of_window << wbits; // each bucket module contains 1<<c buckets; total is num_of_window << c
        // printf("bucket_method_msm, bitsize: %d, c: %d, num_of_window: %d, total_num_of_buckets:%d \n", bitsize, c, num_of_window, total_num_of_buckets);
        CUDA_OK(cudaMallocAsync(&buckets, sizeof(P) * total_num_of_buckets, gpu));

        // launch the bucket initialization kernel with maximum threads
        unsigned NUM_THREADS = 1 << 10; // NOTE: can tune this for better performance;
        unsigned NUM_BLOCKS = (total_num_of_buckets + NUM_THREADS - 1) / NUM_THREADS;
        initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(buckets, total_num_of_buckets);

        // indicate which bucket the scalar belongs to within each window; hence the total size is size*num_of_window;
        // additional size is allocated in the front for the following sorting purpose
        unsigned *scalars_window_bucket_indices;
        // binded to scalars scalars_window_bucket_indices
        unsigned *points_window_bucket_indices;

        CUDA_OK(cudaMallocAsync(&scalars_window_bucket_indices, sizeof(unsigned) * size * (num_of_window + 1), gpu));
        CUDA_OK(cudaMallocAsync(&points_window_bucket_indices, sizeof(unsigned) * size * (num_of_window + 1), gpu));

        // split scalars into digits
        // printf("initialize split_scalars_kernel: size: %d, msm_log_size: %d, num_of_window: %d, bm_bitsize: %d \n", size, msm_log_size, num_of_window, bm_bitsize);
        if (mont)
        {
            transfrom_scalars_and_points_from_mont<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, gpu>>>(d_scalars, d_points, size);
        }

        NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;
        // `scalars_window_bucket_indices` & `points_window_bucket_indices` with offset `size`: leaving the first window free for the out of place sort later
        split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(scalars_window_bucket_indices + size, points_window_bucket_indices + size, d_scalars, size, msm_log_size, num_of_window, num_of_windows_bitsize, wbits);

        // sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
        unsigned *sort_indices_temp_storage{};
        size_t sort_indices_temp_storage_bytes;
        // Call the function initially with a NULL pointer to determine the required temporary storage size
        cub::DeviceRadixSort::SortPairs(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, scalars_window_bucket_indices + size, scalars_window_bucket_indices,
            points_window_bucket_indices + size, points_window_bucket_indices, size, 0, sizeof(unsigned) * 8, gpu);
        CUDA_OK(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, gpu));
        for (unsigned i = 0; i < num_of_window; i++)
        {
            unsigned offset_out = i * size;
            unsigned offset_in = offset_out + size;
            // sort the pair within each window and put the result in previous window;
            // the third last parameter is `begin_bit`, [optional] The least-significant bit index (inclusive) needed for key comparison
            // the second last parameter is `end_bit`, [optional] The most-significant bit index (exclusive) needed for key comparison (e.g., sizeof(unsigned int) * 8)
            cub::DeviceRadixSort::SortPairs(
                sort_indices_temp_storage, sort_indices_temp_storage_bytes, scalars_window_bucket_indices + offset_in,
                scalars_window_bucket_indices + offset_out, points_window_bucket_indices + offset_in, points_window_bucket_indices + offset_out, size, 0, sizeof(unsigned) * 8,
                gpu);
        }
        CUDA_OK(cudaFreeAsync(sort_indices_temp_storage, gpu));

        // stores the unique bucket indices;
        // [  0|0, 0|1, 0|2, ..., 0|(1<<c -1), \
        //    1|0, 1|1, 1|2, ..., 1|(1<<c -1), \
        // ...
        // ]
        unsigned *d_unique_bucket_indices;
        unsigned *d_bucket_count;
        unsigned *d_nof_buckets_to_compute;
        CUDA_OK(cudaMallocAsync(&d_unique_bucket_indices, sizeof(unsigned) * total_num_of_buckets, gpu));
        CUDA_OK(cudaMallocAsync(&d_bucket_count, sizeof(unsigned) * total_num_of_buckets, gpu));
        CUDA_OK(cudaMallocAsync(&d_nof_buckets_to_compute, sizeof(unsigned), gpu));
        unsigned *encode_temp_storage{};
        size_t encode_temp_storage_bytes = 0;
        // It identifies consecutive repeated elements (runs) in an input array and outputs unique elements along with the count of occurrences for each run
        cub::DeviceRunLengthEncode::Encode(
            encode_temp_storage, encode_temp_storage_bytes, scalars_window_bucket_indices, d_unique_bucket_indices, d_bucket_count,
            d_nof_buckets_to_compute, num_of_window * size, gpu);
        CUDA_OK(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, gpu));
        cub::DeviceRunLengthEncode::Encode(
            encode_temp_storage, encode_temp_storage_bytes, scalars_window_bucket_indices, d_unique_bucket_indices, d_bucket_count,
            d_nof_buckets_to_compute, num_of_window * size, gpu);
        CUDA_OK(cudaFreeAsync(encode_temp_storage, gpu));

        unsigned *d_bucket_offsets; // stores where each new bucket begin; new means new `unique`
        CUDA_OK(cudaMalloc(&d_bucket_offsets, sizeof(unsigned) * total_num_of_buckets));
        unsigned *offsets_temp_storage{};
        size_t offsets_temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            offsets_temp_storage, offsets_temp_storage_bytes, d_bucket_count, d_bucket_offsets, total_num_of_buckets, gpu);
        CUDA_OK(cudaMalloc(&offsets_temp_storage, offsets_temp_storage_bytes));
        cub::DeviceScan::ExclusiveSum(
            offsets_temp_storage, offsets_temp_storage_bytes, d_bucket_count, d_bucket_offsets, total_num_of_buckets, gpu);
        CUDA_OK(cudaFreeAsync(offsets_temp_storage, gpu));

        unsigned h_nof_buckets_to_compute;
        CUDA_OK(cudaMemcpyAsync(&h_nof_buckets_to_compute, d_nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, gpu));

        // if all points are 0 just return point 0
        if (h_nof_buckets_to_compute == 0)
        {
            if (!on_device)
                final_result[0] = P::zero();
            else
            {
                P *h_final_result = (P *)malloc(sizeof(P));
                h_final_result[0] = P::zero();
                CUDA_OK(cudaMemcpyAsync(final_result, h_final_result, sizeof(P), cudaMemcpyHostToDevice, gpu));
            }

            return;
        }

        // sort by bucket sizes
        unsigned *d_sorted_bucket_count;
        CUDA_OK(cudaMallocAsync(&d_sorted_bucket_count, sizeof(unsigned) * h_nof_buckets_to_compute, gpu));
        unsigned *d_sorted_bucket_offsets;
        CUDA_OK(cudaMallocAsync(&d_sorted_bucket_offsets, sizeof(unsigned) * h_nof_buckets_to_compute, gpu));
        unsigned *sort_offsets_temp_storage{};
        size_t sort_offsets_temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_bucket_offsets,
            d_sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, gpu);
        CUDA_OK(cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, gpu));
        cub::DeviceRadixSort::SortPairsDescending(
            sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_bucket_offsets,
            d_sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, gpu);
        CUDA_OK(cudaFreeAsync(sort_offsets_temp_storage, gpu));

        unsigned *d_sorted_unique_bucket_indices;
        CUDA_OK(cudaMallocAsync(&d_sorted_unique_bucket_indices, sizeof(unsigned) * h_nof_buckets_to_compute, gpu));
        unsigned *sort_single_temp_storage{};
        size_t sort_single_temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            sort_single_temp_storage, sort_single_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_unique_bucket_indices,
            d_sorted_unique_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, gpu);
        CUDA_OK(cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, gpu));
        cub::DeviceRadixSort::SortPairsDescending(
            sort_single_temp_storage, sort_single_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_unique_bucket_indices,
            d_sorted_unique_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, gpu);
        CUDA_OK(cudaFreeAsync(sort_single_temp_storage, gpu));

        // find large buckets
        unsigned avarage_size = size / (1 << wbits);
        unsigned bucket_th = large_bucket_factor * avarage_size;
        unsigned *nof_large_buckets;
        CUDA_OK(cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), gpu));
        CUDA_OK(cudaMemset(nof_large_buckets, 0, sizeof(unsigned)));

        unsigned TOTAL_THREADS = 129000; // TODO - device dependant
        unsigned cutoff_run_length = max(2, h_nof_buckets_to_compute / TOTAL_THREADS);
        unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length - 1) / cutoff_run_length;

        NUM_THREADS = min(1 << 5, cutoff_nof_runs);
        NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;
        // printf("h_nof_buckets_to_compute: %d, cutoff_run_length: %d, cutoff_nof_runs: %d, NUM_BLOCKS: %d, NUM_THREADS: %d bucket_th: %d\n", h_nof_buckets_to_compute, cutoff_run_length, cutoff_nof_runs,
        //        NUM_BLOCKS, NUM_THREADS, bucket_th);
        find_cutoff_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(
            d_sorted_bucket_count, h_nof_buckets_to_compute, bucket_th, cutoff_run_length, nof_large_buckets);

        unsigned h_nof_large_buckets;
        CUDA_OK(cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, gpu));

        unsigned h_max_res[2];
        CUDA_OK(cudaMemcpyAsync(&h_max_res[0], d_sorted_bucket_count, sizeof(unsigned), cudaMemcpyDeviceToHost, gpu));
        CUDA_OK(cudaMemcpyAsync(&h_max_res[1], d_sorted_unique_bucket_indices, sizeof(unsigned), cudaMemcpyDeviceToHost, gpu));
        unsigned h_largest_bucket_count = h_max_res[0];
        unsigned h_nof_zero_large_buckets = h_max_res[1];
        unsigned large_buckets_to_compute =
            h_nof_large_buckets > h_nof_zero_large_buckets ? h_nof_large_buckets - h_nof_zero_large_buckets : 0;

        P *large_buckets;

        // printf("large_buckets_to_compute: %d, bucket_th: %d \n", large_buckets_to_compute, bucket_th);
        if (large_buckets_to_compute > 0 && bucket_th > 0)
        {
            unsigned threads_per_bucket = min(MAX_TH,
                                              1 << (unsigned)ceil(log2((h_largest_bucket_count + bucket_th - 1) / bucket_th)));
            unsigned max_bucket_size_run_length = (h_largest_bucket_count + threads_per_bucket - 1) / threads_per_bucket;
            unsigned total_large_buckets_size = large_buckets_to_compute * threads_per_bucket;
            // printf("sizeof(P): %d, total_large_buckets_size: %d\n", sizeof(P), total_large_buckets_size);
            CUDA_OK(cudaMallocAsync(&large_buckets, sizeof(P) * total_large_buckets_size, gpu[2]));

            NUM_THREADS = min(1 << 8, total_large_buckets_size);
            NUM_BLOCKS = (total_large_buckets_size + NUM_THREADS - 1) / NUM_THREADS;
            // printf("NUM_BLOCKS: %d, NUM_THREADS: %d, h_nof_zero_large_buckets: %d, total_num_of_buckets: %d, max_bucket_size_run_length: %d\n",
            // NUM_BLOCKS, NUM_THREADS, h_nof_zero_large_buckets, total_num_of_buckets, max_bucket_size_run_length);
            accumulate_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu[2]>>>(
                large_buckets, d_sorted_bucket_offsets + h_nof_zero_large_buckets, d_sorted_bucket_count + h_nof_zero_large_buckets,
                d_sorted_unique_bucket_indices + h_nof_zero_large_buckets, points_window_bucket_indices, d_points, total_num_of_buckets,
                large_buckets_to_compute, wbits + num_of_windows_bitsize, wbits, threads_per_bucket, max_bucket_size_run_length);

            // reduce
            for (int s = total_large_buckets_size >> 1; s > large_buckets_to_compute - 1; s >>= 1)
            {
                NUM_THREADS = min(MAX_TH, s);
                NUM_BLOCKS = (s + NUM_THREADS - 1) / NUM_THREADS;
                single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu[2]>>>(
                    large_buckets, large_buckets, s * 2, 0, 0, 0, s);

                //   CHECK_LAST_CUDA_ERROR();
            }

            // distribute
            NUM_THREADS = min(MAX_TH, large_buckets_to_compute);
            NUM_BLOCKS = (large_buckets_to_compute + NUM_THREADS - 1) / NUM_THREADS;
            distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu[2]>>>(
                large_buckets, buckets, d_sorted_unique_bucket_indices + h_nof_zero_large_buckets, large_buckets_to_compute);
        }
        else
        {
            h_nof_large_buckets = 0; // if bucket_th = 0; there is no large buckets
        }

        // launch the accumulation kernel with maximum threads
        if (h_nof_buckets_to_compute > h_nof_large_buckets)
        {
            NUM_THREADS = 1 << 8;
            NUM_BLOCKS = (h_nof_buckets_to_compute - h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
            accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(
                buckets, d_sorted_bucket_offsets + h_nof_large_buckets, d_sorted_bucket_count + h_nof_large_buckets,
                d_sorted_unique_bucket_indices + h_nof_large_buckets, points_window_bucket_indices, d_points, total_num_of_buckets,
                h_nof_buckets_to_compute - h_nof_large_buckets, wbits + num_of_windows_bitsize, wbits);
        }
        // CUDA_OK(cudaFreeAsync(large_buckets, stream2));
        // all the large buckets need to be accumulated before the final summation
        CUDA_OK(cudaStreamSynchronize(gpu[2]));
        CUDA_OK(cudaStreamDestroy(gpu[2]));

        P *d_final_result;
        if (!on_device)
            CUDA_OK(cudaMallocAsync(&d_final_result, sizeof(P), gpu));

        P *final_results;
        if (big_triangle)
        {
            CUDA_OK(cudaMallocAsync(&final_results, sizeof(P) * num_of_window, gpu));
            // launch the bucket module sum kernel - a thread for each bucket module
            NUM_THREADS = num_of_window;
            NUM_BLOCKS = 1;

            big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(buckets, final_results, num_of_window, wbits);
        }
        else
        {
            unsigned source_bits_count = wbits;
            unsigned source_windows_count = num_of_window;
            unsigned source_buckets_count = total_num_of_buckets;
            P *source_buckets = buckets;
            buckets = nullptr;
            P *target_buckets;
            P *temp_buckets1;
            P *temp_buckets2;
            /**
             * Assuming C is 4, which represents a window size of 4 bits, and target_c is 2, indicating a target window size of 2 bits.

                Suppose source_windows_count is 64, signifying that there are 64 source windows. In this case, target_windows_count would be 128.

                If source_buckets is an array with a length of 1024:

                In the first round:
                The first call to single_stage_multi_reduction_kernel is made with nof_threads set to 512.
                The launch parameters are (2, 1, 1) and (256, 1, 1).
                The kernel parameters are (source_buckets, temp_buckets1, 16, 0, 0, 0, 512).
                Here, the "jump" variable is 8, and the operation is as follows:

                temp_buckets1[0] = source_buckets[0] + source_buckets[8]
                temp_buckets1[1] = source_buckets[1] + source_buckets[9]
                temp_buckets1[2] = source_buckets[2] + source_buckets[10]
                ...
                temp_buckets1[7] = source_buckets[7] + source_buckets[15]
                temp_buckets1[8] = source_buckets[16] + source_buckets[24]
                ...
                Ultimately, temp_buckets1 consists of 512 elements, where each element is obtained by taking pairs of elements from source_buckets at specific intervals and adding them together.
                In the second call to single_stage_multi_reduction_kernel, nof_threads remains at 512.
                The launch parameters are once again (2, 1, 1) and (256, 1, 1).
                The kernel parameters are (source_buckets, temp_buckets2, 4, 0, 0, 0, 512).
                This time, the "jump" variable is 2, and the operation is as follows:

                temp_buckets2[0] = source_buckets[0] + source_buckets[2]
                temp_buckets2[1] = source_buckets[1] + source_buckets[3]
                temp_buckets2[2] = source_buckets[4] + source_buckets[6]
                temp_buckets2[3] = source_buckets[5] + source_buckets[7]
                ...
                In the end, temp_buckets2 consists of 512 elements, and, similar to the first call, it pairs up elements from source_buckets at specific intervals and adds them together.
                It's important to note that these two calls do not handle low and high bits separately. Instead, they both involve adding pairs of elements with a specific step size from the source_buckets.
            */
            for (unsigned i = 0;; i++)
            {
                const unsigned target_bits_count = (source_bits_count + 1) >> 1;                     // c/2=8
                const unsigned target_windows_count = source_windows_count << 1;                     // nof bms*2 = 32
                const unsigned target_buckets_count = target_windows_count << target_bits_count;     // bms*2^c = 32*2^8
                CUDA_OK(cudaMallocAsync(&target_buckets, sizeof(P) * target_buckets_count, gpu));    // 32*2^8*2^7 buckets
                CUDA_OK(cudaMallocAsync(&temp_buckets1, sizeof(P) * source_buckets_count / 2, gpu)); // 32*2^8*2^7 buckets
                CUDA_OK(cudaMallocAsync(&temp_buckets2, sizeof(P) * source_buckets_count / 2, gpu)); // 32*2^8*2^7 buckets

                if (source_bits_count > 0)
                {
                    for (unsigned j = 0; j < target_bits_count; j++)
                    {
                        /**
                         * there are two calls to single_stage_multi_reduction_kernel
                         * it splits windows of bitsize c into windows of bitsize c/2, and the first call computes the lower window and the second computes higher.
                         * An example for c=4, it looks like:
                         * 1*P_1 + 2*P_2 + ... + 15*P_15 = ((P_1 + P_2 + P_9 + P_13) + ... + 3*(P_3 + P_7 + P_11 + P_15)) + 4*((P_4 + P_5 + P_6 + P_7) + ... + 3*(P_12 + P_13 + P_14 + P_15))
                         * The first sum is the first single_stage_multi_reduction_kernel and the second sum (that's multiplied by 4) is the second call
                         */
                        unsigned nof_threads = (source_buckets_count >> (1 + j));
                        NUM_THREADS = min(MAX_TH, nof_threads);
                        NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
                        single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(
                            j == 0 ? source_buckets : temp_buckets1, j == target_bits_count - 1 ? target_buckets : temp_buckets1,
                            1 << (source_bits_count - j), j == target_bits_count - 1 ? 1 << target_bits_count : 0, 0, 0, nof_threads);

                        NUM_THREADS = min(MAX_TH, nof_threads);
                        NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
                        single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(
                            j == 0 ? source_buckets : temp_buckets2, j == target_bits_count - 1 ? target_buckets : temp_buckets2,
                            1 << (target_bits_count - j), j == target_bits_count - 1 ? 1 << target_bits_count : 0, 1, 0, nof_threads);
                    }
                }
                if (target_bits_count == 1)
                {
                    num_of_window = scalar_t::nbits;
                    CUDA_OK(cudaMallocAsync(&final_results, sizeof(P) * num_of_window, gpu));
                    NUM_THREADS = 32;
                    NUM_BLOCKS = (num_of_window + NUM_THREADS - 1) / NUM_THREADS;
                    last_pass_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, gpu>>>(target_buckets, final_results, num_of_window);
                    wbits = 1;
                    CUDA_OK(cudaFreeAsync(source_buckets, gpu));
                    CUDA_OK(cudaFreeAsync(target_buckets, gpu));
                    CUDA_OK(cudaFreeAsync(temp_buckets1, gpu));
                    CUDA_OK(cudaFreeAsync(temp_buckets2, gpu));
                    break;
                }
                CUDA_OK(cudaFreeAsync(source_buckets, gpu));
                CUDA_OK(cudaFreeAsync(temp_buckets1, gpu));
                CUDA_OK(cudaFreeAsync(temp_buckets2, gpu));
                source_buckets = target_buckets;
                target_buckets = nullptr;
                temp_buckets1 = nullptr;
                temp_buckets2 = nullptr;
                source_bits_count = target_bits_count;
                source_windows_count = target_windows_count;
                source_buckets_count = target_buckets_count;
            }
        }

        // printf("double and add \n");

        // launch the double and add kernel, a single thread
        final_accumulation_kernel<P, S>
            <<<1, 1, 0, gpu>>>(final_results, on_device ? final_result : d_final_result, 1, num_of_window, wbits);
        //  printf("free \n");

        CUDA_OK(cudaFreeAsync(final_results, gpu));
        CUDA_OK(cudaStreamSynchronize(gpu));

        if (!on_device)
            CUDA_OK(cudaMemcpyAsync(final_result, d_final_result, sizeof(P), cudaMemcpyDeviceToHost, gpu));

        // free memory
        if (!on_device)
        {
            CUDA_OK(cudaFreeAsync(d_points, gpu));
            CUDA_OK(cudaFreeAsync(d_scalars, gpu));
            CUDA_OK(cudaFreeAsync(d_final_result, gpu));
        }
        CUDA_OK(cudaFreeAsync(buckets, gpu));

        CUDA_OK(cudaFreeAsync(scalars_window_bucket_indices, gpu));
        CUDA_OK(cudaFreeAsync(points_window_bucket_indices, gpu));
        CUDA_OK(cudaFreeAsync(d_unique_bucket_indices, gpu));
        CUDA_OK(cudaFreeAsync(d_bucket_count, gpu));
        CUDA_OK(cudaFreeAsync(d_nof_buckets_to_compute, gpu));
        CUDA_OK(cudaFreeAsync(d_bucket_offsets, gpu));

        CUDA_OK(cudaFreeAsync(d_sorted_bucket_count, gpu));
        CUDA_OK(cudaFreeAsync(d_sorted_bucket_offsets, gpu));
        CUDA_OK(cudaFreeAsync(d_sorted_unique_bucket_indices, gpu));
        CUDA_OK(cudaFreeAsync(nof_large_buckets, gpu));
        //   if (large_buckets_to_compute > 0 && bucket_th > 0) cudaFreeAsync(large_buckets, stream);
        CUDA_OK(cudaStreamSynchronize(gpu));
        CUDA_OK(cudaStreamDestroy(gpu));
    }


};

/**
 * MSM on G1 curve using pippenger algorithm
 * @param out, MSM result; defined in Jacobian format; type is `point_t`, it contains X,Y,Z
 * @param points, the input points; defined in Affine format type is `affine_t`, it contains X,Y
 * @param scalars, the input scalars; Fr scalar field on the pairing curve
 * @param mont, specify whether the input scalars are in montgomery format
 *
 * @class bucket_t, defined as `xyzz_t` the extended jacobian format; contains X,Y,ZZ,ZZZ
 */
template <class point_t, class affine_t, class scalar_t>
static RustError mult_pippenger(point_t *out, affine_t *points, size_t npoints,
                                scalar_t *scalars, bool mont,
                                bool on_device,
                                bool big_triangle,
                                unsigned large_bucket_factor)
{
    try
    {
        msm_t<point_t, affine_t, scalar_t> msm{points, scalars, npoints};
        // msm.bucket_method_msm(*out, slice_t<affine_t>{points, npoints},
        //                                 slice_t<affine_t>{scalars, npoints}, npoints, mont, on_device, big_triangle, large_bucket_factor);
        return RustError{0};
    }
    catch (const cuda_error &e)
    {
        return RustError{e.code(), e.what()};
    }
}
#endif
