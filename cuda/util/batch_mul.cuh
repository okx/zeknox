#ifndef UTILS_KERNELS_H
#define UTILS_KERNELS_H

#define MAX_THREADS_PER_BLOCK 256

// namespace utils_internal
// {
  template <typename E, typename S>
  int batch_vector_mult(S *scalar_vec, E *element_vec, unsigned n_scalars, unsigned batch_size, cudaStream_t stream)
  {
    // Set the grid and block dimensions
    int NUM_THREADS = MAX_THREADS_PER_BLOCK;
    int NUM_BLOCKS = (n_scalars * batch_size + NUM_THREADS - 1) / NUM_THREADS;
    batchVectorMult<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(scalar_vec, element_vec, n_scalars, batch_size);
    return 0;
  }

  template <typename E, typename S>
  __global__ void batchVectorMult(S *scalar_vec, E *element_vec, unsigned n_scalars, unsigned batch_size)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_scalars * batch_size)
    {
      int scalar_id = tid % n_scalars;
      element_vec[tid] = scalar_vec[scalar_id] * element_vec[tid];
    }
  }

// } // namespace utils_internal

#endif