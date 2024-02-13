```sh
cmake .. -DTARGET_PLATFORM=macos_arm64 
```

```cpp
set_source_files_properties(cuda/ntt/parameters.cuh PROPERTIES LANGUAGE CUDA)
 set_target_properties(cryptography_cuda
         PROPERTIES
                 CUDA_RUNTIME_LIBRARY Shared
                 # CUDA_STANDARD 14 # this one cannot be changed by CMake
                 # CUDA_SEPARABLE_COMPILATION ON # not needed for this example
 )
 # # # set_property(TARGET CUDA_COMP PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)

set_target_properties(cryptography_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")
# find_library(CUDA_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# target_compile_features(cryptography_cuda PRIVATE cuda_std_14)
```

- `CUDA_SEPARABLE_COMPILATION`
```sh
find_package(CUDAToolkit REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
```
 Separable compilation allows the compilation of individual CUDA source files (*.cu) into separate object files, which can be linked together later. This can result in faster incremental builds, as changes to one CUDA source file may only require recompilation of that specific file.