nvcc predefines the following macros:
- __NVCC__ (1): Defined when compiling C/C++/CUDA source files.
- __CUDACC__ (1): Defined when compiling CUDA source files.

# compile a single cuda file
```
nvcc -gencode arch=compute_70,code=sm_70 -g -G kernel.cu -o kernel
```

# references
- [nvcc compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#)
