#if defined(__NVCC__) && !defined(__CUDA_ARCH__)
#include <atomic>


// extern "C"
// bool cuda_available()
// {
    
//     static std::atomic<int> available(0);
//     int ret = available;
   
//     cudaDeviceProp prop;
//     ret = cudaGetDeviceProperties(&prop, 0) == cudaSuccess &&
//             prop.major >= 7;
//     printf("gpu name: %s , major: %d, minor: %d \n", prop.name, prop.major, prop.minor);
//     available = ret;
//     return (bool)ret;
// }
#endif
