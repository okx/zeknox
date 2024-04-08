#ifndef __HIP_UTILS_HIP__
#define __HIP_UTILS_HIP__

#include "hip/hip_runtime.h"
#include <iostream>

constexpr int error_exit_code = -1;

#define HIP_CHECK(condition)                                                                \
{                                                                                       \
    const hipError_t error = condition;                                                 \
    if(error != hipSuccess)                                                             \
    {                                                                                   \
        std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                  << __FILE__ << ':' << __LINE__ << std::endl;                          \
        std::exit(error_exit_code);                                                     \
    }                                                                                   \
}

#endif      // __HIP_UTILS_HIP__