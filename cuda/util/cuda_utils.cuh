#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cuda.h>
#include <stdio.h>

#ifdef USE_CUDA

#define INLINE __device__ __forceinline__
#define CONST __device__
#define DEVICE __device__
#define LOCATION gpu

#else // USE_CUDA

#define INLINE __host__ inline
#define CONST const
#define DEVICE __host__
#define LOCATION host

#endif // USE_CUDA

#define PASTER(x,y) x ## _ ## y
#define EVALUATOR(x,y)  PASTER(x,y)
#define FUNC(name) EVALUATOR(LOCATION, name)
#define VAR(name) EVALUATOR(LOCATION, name)

__host__ inline void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        printf("CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

#define CHECKCUDAERR(ans)                          \
    {                                              \
        checkCudaError((ans), __FILE__, __LINE__); \
    }

#endif  // __CUDA_UTILS_H__