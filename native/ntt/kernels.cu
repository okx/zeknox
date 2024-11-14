// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_CUDA_NTTT_KERNELS_CU__
#define __ZEKNOX_CUDA_NTTT_KERNELS_CU__

#include <cooperative_groups.h>
#include <utils/sharedmem.cuh>
#include <curand_kernel.h>

#define BLOCK_DIM 32

/**
 * Bit reversal perumuation of an array. See : https://en.wikipedia.org/wiki/Bit-reversal_permutation
 */
__global__ void reverse_order_kernel(fr_t *arr, uint32_t n, uint32_t logn, uint32_t batch_size)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadId < n * batch_size)
    {
        int idx = threadId % n;
        int batch_idx = threadId / n;
        int idx_reversed = __brev(idx) >> (32 - logn);

        // Check to ensure that the larger index swaps with the smaller one
        if (idx > idx_reversed)
        {
            // Swap with temp
            fr_t temp = arr[batch_idx * n + idx];
            arr[batch_idx * n + idx] = arr[batch_idx * n + idx_reversed];
            arr[batch_idx * n + idx_reversed] = temp;
        }
    }
}

/**
 * Extends the batch of polynomials with padded 0 values (Note this polynomial is in point-value form).
 * We append each polynomial (represented as a row in the matrix) with a
 * set of 0's till the array is of length n_extended. (n_extended columns)
 *
 * m, n before extension
 * m, n_extended after extension
 */
__global__ void degree_extension_kernel(fr_t *output, fr_t *input, uint32_t n, uint32_t n_extend, uint32_t batch_size)
{
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (threadId < n_extend * batch_size)
    {
        int idx = threadId % n_extend;
        int batch_idx = threadId / n_extend;

        // Is not an extended input
        if (idx < n)
        {

            output[batch_idx * n_extend + idx] = input[batch_idx * n + idx];
        }
        else
        {
            output[batch_idx * n_extend + idx] = fr_t::zero();
        }
    }
}

__global__ void gen_random_salt_kernel(fr_t *arr, uint32_t size, uint32_t salt_size, uint64_t seed)
{
#if defined(FEATURE_GOLDILOCKS)
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid >= size)
    {
        return;
    }

    curandState rstate;
    curand_init(seed, tid, 0, &rstate);

    for (uint32_t k = 0; k < salt_size; k++)
    {
        uint32_t x1 = curand(&rstate);
        uint32_t x2 = curand(&rstate);
        uint64_t x = x1;
        x = (x << 32) | x2;
        arr[k * size + tid] = fr_t(x);
    }
#endif
}

/**
 * Fast transpose from NVIDIA found at https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ along by a bit reversal permutation of the transposed matrix.
 * Takes n x m matrix and returns its transposed m x n matrix with the values being bit reversed.
 * We use shared memory 'cache' blocks for coalesce memory efficiency improvement
 * use dynamically allocated shared memory: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/and https://stackoverflow.com/questions/69963358/how-to-fix-warning-dynamic-initialization-is-not-supported-for-a-function-scop
 * @param in_arr input array of type E (elements).
 * @param out_arr output array of type E (elements).
 * @param n column size of in_arr.
 * @param batch_size row size of in_arr.
 */
__global__ void transpose_rev_kernel(fr_t *in_arr, fr_t *out_arr, uint32_t n, uint32_t lg_n, uint32_t batch_size)
{
    extern __shared__ fr_t block[];

    // Get indexes
    int j_idx = blockIdx.x * BLOCK_DIM + (4 * threadIdx.x);
    int i_idx = blockIdx.y * BLOCK_DIM + threadIdx.y;

    int idx = i_idx * n + j_idx;

    // read the matrix tile into shared memory in its transposed position
    for (int i = 0; i < 4; i++)
    {
        if ((i_idx < batch_size) && ((j_idx + i) < n))
        {
            block[((4 * threadIdx.x) + i) * BLOCK_DIM + threadIdx.y] = in_arr[idx + i];
        }
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // calculated transposed indexes
    j_idx = blockIdx.y * BLOCK_DIM + (4 * threadIdx.x);
    i_idx = blockIdx.x * BLOCK_DIM + threadIdx.y;

    int i_idx_rev = __brev(i_idx) >> (32 - lg_n);

    idx = i_idx_rev * batch_size + j_idx;

    // write the transposed matrix tile to global memory (out_arr) in linear order
    for (int i = 0; i < 4; i++)
    {
        if ((i_idx < n) && ((j_idx + i) < batch_size))
        {
            out_arr[idx + i] = block[threadIdx.y * BLOCK_DIM + 4 * threadIdx.x + i];
        }
    }
}

__global__ void twiddle_factors_kernel(fr_t *d_twiddles, uint32_t n_twiddles, fr_t omega)
{
    for (uint32_t i = 0; i < n_twiddles; i++)
    {
        d_twiddles[i] = fr_t::zero();
    }
    d_twiddles[0] = fr_t::one();
    for (uint32_t i = 0; i < n_twiddles - 1; i++)
    {
        d_twiddles[i + 1] = omega * d_twiddles[i];
    }
}

/**
 * Cooley-Tukey NTT implementation, see: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm .
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of twiddles.
 * @param max_task max count of parallel tasks.
 * @param s log2(n) loop index.
 */
__global__ void
ntt_template_kernel(fr_t *arr, uint32_t n, fr_t *twiddles, uint32_t n_twiddles, uint32_t max_task, uint32_t s, bool rev)
{
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    // printf("inside ntt_template_kernel blk.idx: %d, thread.idx: %d\n", blockIdx.x, threadIdx.x );
    // }

    int task = blockIdx.x;
    int chunks = n / (blockDim.x * 2); // how many chunks within one NTT

    if (task < max_task)
    {
        // flattened loop allows parallel processing
        uint32_t l = threadIdx.x;
        uint32_t loop_limit = blockDim.x;

        if (l < loop_limit)
        {
            uint32_t ntw_i = task % chunks; // chunk index of the current NTT

            uint32_t shift_s = 1 << s;        // offset to j,
            uint32_t shift2_s = 1 << (s + 1); // num of continuous elements access
            uint32_t n_twiddles_div = n_twiddles >> (s + 1);

            l = ntw_i * blockDim.x + l; // to l from chunks to full

            uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
            uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
            uint32_t k = i + j + shift_s;

            fr_t tw = twiddles[j * n_twiddles_div];

            uint32_t offset = (task / chunks) * n;
            fr_t u = *(arr + offset + i + j);
            fr_t v = *(arr + offset + k);

            if (!rev)
                v = tw * v;
            *(arr + offset + i + j) = u + v;
            v = u - v;
            if (rev)
            {

                *(arr + offset + k) = tw * v;
            }
            else
            {
                *(arr + offset + k) = v;
            }
        }
    }
}

/**
 * Multiply the elements of an input array by a scalar in-place.
 * @param arr input array.
 * @param n size of arr.
 * @param n_inv scalar of type S (scalar).
 */
__global__ void template_normalize_kernel(fr_t *arr, uint32_t n, fr_t n_inv)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n)
    {
        arr[tid] = n_inv * arr[tid];
    }
}

/**
 * Cooley-Tukey NTT implementation, see: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm .
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of twiddles.
 * @param max_task max count of parallel tasks.
 * @param s log2(n) loop index.
 */
__global__ void ntt_template_kernel_shared_rev(
    fr_t *arr_g,
    uint32_t n,
    const fr_t *r_twiddles,
    uint32_t n_twiddles,
    uint32_t max_task,
    uint32_t ss,
    uint32_t logn,
    int gpu_id)
{
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    // {
    //     printf("inside ntt_template_kernel_shared_rev, before smGetPointer gpu_id: %d>>>>>\n", gpu_id);
    // }
    SharedMemory<fr_t> smem;
    fr_t *arr = smem.getPointer();

    uint32_t task = blockIdx.x;
    uint32_t loop_limit = blockDim.x;
    uint32_t chunks = n / (loop_limit * 2);
    uint32_t offset = (task / chunks) * n;
    if (task < max_task)
    {
        // flattened loop allows parallel processing
        uint32_t l = threadIdx.x;

        if (l < loop_limit)
        {

#pragma unroll
            for (; ss < logn; ss++)
            {
                int s = logn - ss - 1;
                bool is_beginning = ss == 0;
                bool is_end = ss == (logn - 1);

                uint32_t ntw_i = task % chunks;

                uint32_t n_twiddles_div = n_twiddles >> (s + 1);

                uint32_t shift_s = 1 << s;
                uint32_t shift2_s = 1 << (s + 1);

                l = ntw_i * loop_limit + l; // to l from chunks to full

                uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
                uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
                uint32_t oij = i + j;
                uint32_t k = oij + shift_s;
                // printf("before get twiddle, index: %d, gpu_id: %d\n",j * n_twiddles_div, gpu_id);
                fr_t tw = r_twiddles[j * n_twiddles_div];
                // printf("twiddle of: %d, is: %lu, gpu_id: %d\n", j * n_twiddles_div, tw, gpu_id);
                fr_t u = is_beginning ? arr_g[offset + oij] : arr[oij];
                fr_t v = is_beginning ? arr_g[offset + k] : arr[k];

                if (is_end)
                {
                    arr_g[offset + oij] = u + v;
                    arr_g[offset + k] = tw * (u - v);
                }
                else
                {
                    arr[oij] = u + v;
                    arr[k] = tw * (u - v);
                }

                __syncthreads();
            }
        }
    }
}

/**
 * Cooley-Tuckey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of twiddles.
 * @param max_task max count of parallel tasks.
 * @param s log2(n) loop index.
 */
__global__ void ntt_template_kernel_shared(
    fr_t *__restrict__ arr_g,
    uint32_t n,
    const fr_t *__restrict__ r_twiddles,
    uint32_t n_twiddles,
    uint32_t max_task,
    uint32_t s,
    uint32_t logn)
{
    SharedMemory<fr_t> smem;
    fr_t *arr = smem.getPointer();

    uint32_t task = blockIdx.x;
    uint32_t loop_limit = blockDim.x;
    uint32_t chunks = n / (loop_limit * 2);
    uint32_t offset = (task / chunks) * n;
    if (task < max_task)
    {
        // flattened loop allows parallel processing
        uint32_t l = threadIdx.x;

        if (l < loop_limit)
        {
#pragma unroll
            for (; s < logn; s++) // TODO: this loop also can be unrolled
            {
                uint32_t ntw_i = task % chunks;

                uint32_t n_twiddles_div = n_twiddles >> (s + 1);

                uint32_t shift_s = 1 << s;
                uint32_t shift2_s = 1 << (s + 1);

                l = ntw_i * loop_limit + l; // to l from chunks to full

                uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
                uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
                uint32_t oij = i + j;
                uint32_t k = oij + shift_s;
                fr_t tw = r_twiddles[j * n_twiddles_div];

                fr_t u = s == 0 ? arr_g[offset + oij] : arr[oij];
                fr_t v = s == 0 ? arr_g[offset + k] : arr[k];
                // v = tw * v;
                // if (s == (logn - 1))
                // {
                //     printf("set with offset: offset: %d, oij: %d, k: %d\n", offset, oij, k);
                //     arr_g[offset + oij] = u + tw*v;
                //     arr_g[offset + k] =  u- tw*v;
                // }
                // else
                // {
                //     printf("set without offset\n");
                //     arr[oij] = u + tw*v;
                //     arr[k] = u- tw*v;
                // }
                v = tw * v;
                if (s == (logn - 1))
                {
                    arr_g[offset + oij] = u + v;
                    arr_g[offset + k] = u - v;
                }
                else
                {
                    arr[oij] = u + v;
                    arr[k] = u - v;
                }
                __syncthreads();
            }
        }
    }
}

#endif /**ZEKNOX_CUDA_NTTT_KERNELS_CU_ */
