#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cuda.h>
#include <stdio.h>

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