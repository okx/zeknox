#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>

#ifdef USE_CUDA
#include <cuda.h>

#define INLINE __forceinline__
#define CONST __device__
#define DEVICE __device__
#define LOCATION gpu

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

#else // USE_CUDA

#define INLINE inline
#define CONST const
#define DEVICE
#define LOCATION host

#endif // USE_CUDA

#define PASTER(x,y) x ## _ ## y
#define EVALUATOR(x,y)  PASTER(x,y)
#define FUNC(name) EVALUATOR(LOCATION, name)
#define VAR(name) EVALUATOR(LOCATION, name)



#endif  // __CUDA_UTILS_H__