#ifndef __CRYPTO_MSM_KERNELS_PIPPENGER_CU__
#define __CRYPTO_MSM_KERNELS_PIPPENGER_CU__

__global__ void
find_cutoff_kernel(unsigned* v, unsigned size, unsigned cutoff, unsigned run_length, unsigned* result)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned nof_threads = (size + run_length - 1) / run_length;
  if (tid >= nof_threads) { return; }
  const unsigned start_index = tid * run_length;
  for (int i = start_index; i < min(start_index + run_length, size - 1); i++) {
    if (v[i] > cutoff && v[i + 1] <= cutoff) {
      result[0] = i + 1;
      return;
    }
  }
  if (tid == 0 && v[size - 1] > cutoff) { result[0] = size; }
}

__global__ void
find_max_size(unsigned* bucket_sizes, unsigned* single_bucket_indices, unsigned c, unsigned* largest_bucket_size)
{
  for (int i = 0;; i++) {
    if (single_bucket_indices[i] & ((1 << c) - 1)) {
      largest_bucket_size[0] = bucket_sizes[i];
      largest_bucket_size[1] = i;
      break;
    }
  }
}


// this kernel computes the final result using the double and add algorithm
// it is done by a single thread
template <typename P, typename S>
__global__ void
final_accumulation_kernel(P* final_sums, P* final_results, unsigned nof_msms, unsigned nof_bms, unsigned c)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > nof_msms) return;
  P final_result = P::zero();
  for (unsigned i = nof_bms; i > 1; i--) {
    final_result = final_result + final_sums[i - 1 + tid * nof_bms]; // add
    for (unsigned j = 0; j < c; j++)                                 // double
    {
      final_result = final_result + final_result;
    }
  }
  final_results[tid] = final_result + final_sums[tid * nof_bms];
}

template <typename P, typename A>
__global__ void accumulate_large_buckets_kernel(
  P* __restrict__ buckets,
  unsigned* __restrict__ bucket_offsets,
  unsigned* __restrict__ bucket_sizes,
  unsigned* __restrict__ single_bucket_indices,
  const unsigned* __restrict__ point_indices,
  A* __restrict__ points,
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
  if (tid >= nof_buckets_to_compute * threads_per_bucket) { return; }
  if ((single_bucket_indices[large_bucket_index] & ((1 << c) - 1)) == 0) { // dont need
    return;                                                                // skip zero buckets
  }
  unsigned write_bucket_index = bucket_segment_index * nof_buckets_to_compute + large_bucket_index;
  const unsigned bucket_offset = bucket_offsets[large_bucket_index] + bucket_segment_index * max_run_length;
  const unsigned bucket_size = bucket_sizes[large_bucket_index] > bucket_segment_index * max_run_length
                                 ? bucket_sizes[large_bucket_index] - bucket_segment_index * max_run_length
                                 : 0;
  P bucket;
  unsigned run_length = min(bucket_size, max_run_length);
  for (unsigned i = 0; i < run_length;
       i++) { // add the relevant points starting from the relevant offset up to the bucket size
    unsigned point_ind = point_indices[bucket_offset + i];
    A point = points[point_ind];
    bucket = i ? bucket + point : P::from_affine(point); // init empty buckets
  }
  buckets[write_bucket_index] = run_length ? bucket : P::zero();
}


template <typename P>
__global__ void last_pass_kernel(P* final_buckets, P* final_sums, unsigned num_sums)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid > num_sums) return;
  final_sums[tid] = final_buckets[2 * tid + 1];
}

template <typename P>
__global__ void single_stage_multi_reduction_kernel(
  P* v,
  P* v_r,
  unsigned block_size,
  unsigned write_stride,
  unsigned write_phase,
  unsigned padding,
  unsigned num_of_threads)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_of_threads) { return; }

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

// this kernel sums the entire bucket module
// each thread deals with a single bucket module
template <typename P>
__global__ void big_triangle_sum_kernel(P* buckets, P* final_sums, unsigned nof_bms, unsigned c)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nof_bms) return;
#ifdef SIGNED_DIG
  unsigned buckets_in_bm = (1 << c) + 1;
#else
  unsigned buckets_in_bm = (1 << c);
#endif
  P line_sum = buckets[(tid + 1) * buckets_in_bm - 1];
  final_sums[tid] = line_sum;
  for (unsigned i = buckets_in_bm - 2; i > 0; i--) {
    line_sum = line_sum + buckets[tid * buckets_in_bm + i]; // using the running sum method
    final_sums[tid] = final_sums[tid] + line_sum;
  }
}

// this kernel adds up the points in each bucket
//  __global__ void accumulate_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets,
//   unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, unsigned *__restrict__
//   point_indices, A *__restrict__ points, unsigned nof_buckets, unsigned batch_size, unsigned msm_idx_shift){
template <typename P, typename A>
__global__ void accumulate_buckets_kernel(
  P* __restrict__ buckets,
  unsigned* __restrict__ bucket_offsets,
  unsigned* __restrict__ bucket_sizes,
  unsigned* __restrict__ single_bucket_indices,
  const unsigned* __restrict__ point_indices,
  A* __restrict__ points,
  const unsigned nof_buckets,
  const unsigned nof_buckets_to_compute,
  const unsigned msm_idx_shift,
  const unsigned c)
{
  constexpr unsigned sign_mask = 0x80000000;
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nof_buckets_to_compute) return;
  if ((single_bucket_indices[tid] & ((1 << c) - 1)) == 0) {
    return; // skip zero buckets
  }
#ifdef SIGNED_DIG // todo - fix
  const unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
  const unsigned bm_index = (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1)) >> c;
  const unsigned bucket_index =
    msm_index * nof_buckets + bm_index * ((1 << (c - 1)) + 1) + (single_bucket_indices[tid] & ((1 << c) - 1));
#else
  unsigned msm_index = single_bucket_indices[tid] >> msm_idx_shift;
  unsigned bucket_index = msm_index * nof_buckets + (single_bucket_indices[tid] & ((1 << msm_idx_shift) - 1));
#endif
  const unsigned bucket_offset = bucket_offsets[tid];
  const unsigned bucket_size = bucket_sizes[tid];

  P bucket; // get rid of init buckets? no.. because what about buckets with no points
  for (unsigned i = 0; i < bucket_size;
       i++) { // add the relevant points starting from the relevant offset up to the bucket size
    unsigned point_ind = point_indices[bucket_offset + i];
#ifdef SIGNED_DIG
    unsigned sign = point_ind & sign_mask;
    point_ind &= ~sign_mask;
    A point = points[point_ind];
    if (sign) point = A::neg(point);
#else
    A point = points[point_ind];
#endif
    bucket = i ? bucket + point : P::from_affine(point);
  }
  buckets[bucket_index] = bucket;
}

// this kernel initializes the buckets with zero points
// each thread initializes a different bucket
template <typename P>
__global__ void initialize_buckets_kernel(P* buckets, unsigned N)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) buckets[tid] = P::zero(); // zero point
}

// this kernel splits the scalars into digits of size c
// each thread splits a single scalar into nof_bms digits
template <typename S>
__global__ void split_scalars_kernel(
  unsigned* buckets_indices,
  unsigned* point_indices,
  S* scalars,
  unsigned total_size,
  unsigned msm_log_size,
  unsigned nof_bms,
  unsigned bm_bitsize,
  unsigned c)
{
  constexpr unsigned sign_mask = 0x80000000;
  // constexpr unsigned trash_bucket = 0x80000000;
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index;
  unsigned bucket_index2;
  unsigned current_index;
  unsigned msm_index = tid >> msm_log_size;
  unsigned borrow = 0;
  if (tid < total_size) {
    S scalar = scalars[tid];
    for (unsigned bm = 0; bm < nof_bms; bm++) {
      bucket_index = scalar.get_scalar_digit(bm, c);
#ifdef SIGNED_DIG
      bucket_index += borrow;
      borrow = 0;
      unsigned sign = 0;
      if (bucket_index > (1 << (c - 1))) {
        bucket_index = (1 << c) - bucket_index;
        borrow = 1;
        sign = sign_mask;
      }
#endif
      current_index = bm * total_size + tid;
#ifdef SIGNED_DIG
      point_indices[current_index] = sign | tid; // the point index is saved for later
#else
      buckets_indices[current_index] =
        (msm_index << (c + bm_bitsize)) | (bm << c) |
        bucket_index; // the bucket module number and the msm number are appended at the msbs
      if (scalar == S::zero() || bucket_index == 0) buckets_indices[current_index] = 0; // will be skipped
      point_indices[current_index] = tid; // the point index is saved for later
#endif
    }
  }
}

template <typename P>
__global__ void
distribute_large_buckets_kernel(P* large_buckets, P* buckets, unsigned* single_bucket_indices, unsigned size)
{
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= size) { return; }
  buckets[single_bucket_indices[tid]] = large_buckets[tid];
}


#endif