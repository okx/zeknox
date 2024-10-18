#ifndef __CRYPTO_MSM_KERNELS_PIPPENGER_CU__
#define __CRYPTO_MSM_KERNELS_PIPPENGER_CU__

#include <stdio.h>
#include <msm/msm.h>
#include <msm/pippenger.cuh>

/**
 * @param v, v is assumed to be sorted in descending order
 * @param size, number of elements in v
 * @param run_length, each thread will find for a range of length out of the total size
 */
__global__ void
find_cutoff_kernel(unsigned *v, unsigned size, unsigned cutoff, unsigned run_length, unsigned *result)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned nof_threads = (size + run_length - 1) / run_length;
  if (tid >= nof_threads)
  {
    return;
  }
  const unsigned start_index = tid * run_length;
  for (int i = start_index; i < min(start_index + run_length, size - 1); i++)
  {
    if (v[i] > cutoff && v[i + 1] <= cutoff) // since v is sorted descending wise; the cutoff point is defined as here
    {
      result[0] = i + 1;
      return;
    }
  }
  if (tid == 0 && v[size - 1] > cutoff)
  {
    result[0] = size;
  }
}

// this kernel computes the final result using the double and add algorithm
// it is done by a single thread
template <typename P, typename S>
__global__ void
final_accumulation_kernel(P *final_sums, P *final_results, unsigned nof_msms, unsigned num_of_windows, unsigned c)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > nof_msms)
    return;
  P final_result = P::zero();
  for (unsigned i = num_of_windows; i > 1; i--)
  {
    final_result = final_result + final_sums[i - 1 + tid * num_of_windows]; // add
    for (unsigned j = 0; j < c; j++)                                        // double
    {
      final_result = final_result + final_result;
    }
  }
  final_results[tid] = final_result + final_sums[tid * num_of_windows];
}


/**
 * @param bucket_offsets, stores the index where new unique bucket begins; bucket might contains multiple scalars; it's sorted by counts in descending order
 */
template <typename P, typename A>
__global__ void accumulate_large_buckets_kernel(
    P *__restrict__ buckets,
    unsigned *__restrict__ bucket_offsets,
    unsigned *__restrict__ bucket_count,
    unsigned *__restrict__ unique_bucket_indices,
    const unsigned *__restrict__ point_indices,
    A *__restrict__ points,
    const unsigned nof_buckets,
    const unsigned nof_buckets_to_compute,
    const unsigned msm_idx_shift,
    const unsigned c,
    const unsigned threads_per_bucket,
    const unsigned max_run_length)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned large_bucket_index = tid / threads_per_bucket;
  unsigned bucket_segment_index = tid % threads_per_bucket;
  if (tid >= nof_buckets_to_compute * threads_per_bucket)
  {
    return;
  }
  if ((unique_bucket_indices[large_bucket_index] & ((1 << c) - 1)) == 0) // skip zero buckets
  {
    return;
  }
  unsigned write_bucket_index = bucket_segment_index * nof_buckets_to_compute + large_bucket_index;
  const unsigned bucket_offset = bucket_offsets[large_bucket_index] + bucket_segment_index * max_run_length;
  const unsigned bucket_size = bucket_count[large_bucket_index] > bucket_segment_index * max_run_length
                                   ? bucket_count[large_bucket_index] - bucket_segment_index * max_run_length
                                   : 0;
  P bucket;
  unsigned run_length = min(bucket_size, max_run_length);
  for (unsigned i = 0; i < run_length; i++)
  { // add the relevant points starting from the relevant offset up to the bucket size
    unsigned point_ind = point_indices[bucket_offset + i];
    A point = points[point_ind];
    bucket = i ? bucket + point : P::from_affine(point); // init empty buckets
  }
  buckets[write_bucket_index] = run_length ? bucket : P::zero();
}

template <typename P>
__global__ void last_pass_kernel(P *final_buckets, P *final_sums, unsigned num_sums)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > num_sums)
    return;
  final_sums[tid] = final_buckets[2 * tid + 1];
}

template <typename P>
__global__ void single_stage_multi_reduction_kernel(
    P *v,
    P *v_r,
    unsigned block_size,
    unsigned write_stride,
    unsigned write_phase,
    unsigned padding,
    unsigned num_of_threads)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_of_threads)
  {
    return;
  }

  int jump = block_size / 2;
  int tid_p = padding ? (tid / (2 * padding)) * padding + tid % padding : tid;
  int block_id = tid_p / jump;
  int block_tid = tid_p % jump;
  unsigned read_ind = block_size * block_id + block_tid;
  unsigned write_ind = tid;
  unsigned v_r_key =
      write_stride ? ((write_ind / write_stride) * 2 + write_phase) * write_stride + write_ind % write_stride : write_ind;
  P v_r_value = padding ? (tid % (2 * padding) < padding) ? v[read_ind] + v[read_ind + jump] : P::zero()
                        : v[read_ind] + v[read_ind + jump];

  v_r[v_r_key] = v_r_value;
}

// this kernel sums across windows
// each thread deals with a single window
template <typename P>
__global__ void big_triangle_sum_kernel(P *buckets, P *final_sums, unsigned num_of_windows, unsigned c)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= num_of_windows)
    return;

  unsigned num_of_buckets_per_window = (1 << c);

  P line_sum = buckets[(tid + 1) * num_of_buckets_per_window - 1];
  final_sums[tid] = line_sum;
  for (unsigned i = num_of_buckets_per_window - 2; i > 0; i--)
  {
    line_sum = line_sum + buckets[tid * num_of_buckets_per_window + i];
    final_sums[tid] = final_sums[tid] + line_sum;
  }
}

/**
 * this kernel adds up the points in each bucket
 */
template <typename P, typename A>
__global__ void accumulate_buckets_kernel(
    P *__restrict__ buckets,
    unsigned *__restrict__ bucket_offsets,
    unsigned *__restrict__ bucket_count,
    unsigned *__restrict__ unique_bucket_indices,
    const unsigned *__restrict__ point_indices,
    A *__restrict__ points,
    const unsigned nof_buckets,
    const unsigned nof_buckets_to_compute,
    const unsigned msm_idx_shift,
    const unsigned c)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nof_buckets_to_compute)
    return;
  if ((unique_bucket_indices[tid] & ((1 << c) - 1)) == 0)
  {
    return; // skip zero buckets
  }

  // unsigned msm_index = unique_bucket_indices[tid] >> msm_idx_shift;
  unsigned bucket_index = unique_bucket_indices[tid] & ((1 << msm_idx_shift) - 1);//  msm_index * nof_buckets + ();

  const unsigned bucket_offset = bucket_offsets[tid];
  const unsigned bucket_size = bucket_count[tid];

  P bucket;
  for (unsigned i = 0; i < bucket_size; i++)
  {
    unsigned point_ind = point_indices[bucket_offset + i];
    A point = points[point_ind];
    bucket = i ? bucket + point : P::from_affine(point);
  }
  buckets[bucket_index] = bucket;
}

// this kernel initializes the buckets with zero points
// each thread initializes a different bucket
template <typename P>
__global__ void initialize_buckets_kernel(P *buckets, unsigned N)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N)
    buckets[tid] = P::zero(); // zero point
}

// TODO: try to combine with other kernels. One thread can work on many points
__global__ void transfrom_scalars_and_points_from_mont(scalar_field_t *scalars, g2_affine_t *points, size_t size)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < size)
  {
    *(scalars + tid) = scalar_field_t::from_montgomery(*(scalars + tid));
    *(points + tid) = g2_affine_t::from_montgomery(*(points + tid));
  }
}

/**
 * this kernel splits the scalars into digits of size c; each thread splits a single scalar into num_of_windows digits
 * @param buckets_indices, indicate the bucket index of each scalar of a window. the size of `buckets_indices` is `msm_size`*`num_windows`;the value is the bit concanation of
 * `window_index | window_idx | bucket_index`; concat window_index with bucket_index is required to index all bucket across all windows
 * @param num_windows number of windows; for BN254, the size of bits is 254; if window bit size is `c` = 16; num_windows = ceil(254/16) = 16
 */
template <typename S>
__global__ void split_scalars_kernel(
    unsigned *buckets_indices,
    unsigned *point_indices,
    S *scalars,
    unsigned msm_size,
    unsigned msm_log_size,
    unsigned num_windows,
    unsigned bm_bitsize,
    unsigned c)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index;
  unsigned current_index;
  unsigned window_index = tid >> msm_log_size; // TODO: since tid < msm_size; window_index would always be 0; can remove this.
  if (tid < msm_size)
  {
    S scalar = scalars[tid];
    for (unsigned window_idx = 0; window_idx < num_windows; window_idx++)
    {
      bucket_index = scalar.get_scalar_digit(window_idx, c);
      // printf("tid: %d, window_idx: %d, bucket_index: %d\n", tid, window_idx, bucket_index);
      current_index = window_idx * msm_size + tid;
      buckets_indices[current_index] =
          (window_index << (c + bm_bitsize)) | (window_idx << c) | bucket_index; // the bucket module number and the msm number are appended at the msbs
      if (scalar == S::zero() || bucket_index == 0)
      {
        buckets_indices[current_index] = 0; // will be skipped
      }
      point_indices[current_index] = tid; // the point index is saved for later
    }
  }
}

template <typename P>
__global__ void
distribute_large_buckets_kernel(P *large_buckets, P *buckets, unsigned *unique_bucket_indices, unsigned size)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= size)
  {
    return;
  }
  buckets[unique_bucket_indices[tid]] = large_buckets[tid];
}

#endif