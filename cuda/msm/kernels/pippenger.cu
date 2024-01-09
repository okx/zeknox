#ifndef __CRYPTO_MSM_KERNELS_PIPPENGER_CU__
#define __CRYPTO_MSM_KERNELS_PIPPENGER_CU__

__global__ __forceinline__ void
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

__global__ __forceinline__ void
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


#endif