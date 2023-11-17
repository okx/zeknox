#ifndef __CRYPTO__UTIL_GPU_T_CUH__
#define __CRYPTO__UTIL_GPU_T_CUH__

#ifndef __CUDACC__
# include <cuda_runtime.h>
#endif

#include "slice_t.hpp"

class gpu_t;
size_t ngpus();
const gpu_t& select_gpu(int id = 0);
const cudaDeviceProp& gpu_props(int id = 0);
const std::vector<const gpu_t*>& all_gpus();
extern "C" bool cuda_available();

#endif