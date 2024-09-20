# Description

This repo implements primitives used in Zero Knowledge Proofs accelerated with CUDA for Nvidia GPUs. In particular, this repo is used by OKX Plonky2 [fork](https://github.com/okx/plonky2).

The following primitives are implemented:

- Poseidon hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/poseidon](cuda/poseidon).
- Poseidon hashing over BN254 (or BN128) elements (in C/C++ and CUDA) - see [cuda/poseidon](cuda/poseidon).
- Poseidon2 hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/poseidon2](cuda/poseidon2).
- Keccak hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/keccak](cuda/keccak).
- Monolith hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/monolith](cuda/monolith).
- Merkle Tree building (Plonky2 version) using any of the above hashing methods - see [cuda/merkle](cuda/merkle).
- NTT (including LDE and transpose) over Goldilocks & BN254 elements - see [cuda/ntt](cuda/ntt).
- MSM over Goldilocks & BN254 elements - see [cuda/msm](cuda/msm).

# Building and Testing
## Prerequisites

- git submodules
```
git submodule update --init
```

- nvcc > 12.0 (CUDA Tookit). For the latest CUDA toolkit, please see [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). After installing CUDA, set NVCC environment variable:
```
export NVCC=/usr/local/cuda/bin/nvcc
```

- gcc/g++, make, gtest. To install these in Ubuntu, run:
```
sudo apt install gcc g++ make libgtest-dev
```

## Build CUDA library
```
cd cuda
make lib
```

## Build and run CUDA tests
Note: this requires an Nvidia GPU.

```
cd cuda
make tests.exe
./tests.exe
```

# License

Apache License, Version 2.0 [LICENSE](LICENSE)
