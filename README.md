# description

this repo implements cryptography arithmatic using cuda.

# field to support
- Goldilocks


# compile a single cuda file
```
nvcc -gencode arch=compute_70,code=sm_70 -g -G kernel.cu -o kernel
```